import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from oocr_influence.cli.run_activation_dot_product import ActivationDotProductArgs
from oocr_influence.cli.run_influence import InfluenceArgs, load_influence_scores
from oocr_influence.cli.train_extractive import TrainingArgs
from shared_ml.eval import EvalDataset
from shared_ml.logging import LogState, load_experiment_checkpoint, load_log_from_wandb
from shared_ml.tfidf import get_tfidf_scores
from shared_ml.utils import hash_str

INFLUENCE_SCORES_SCHEMA = Features(
    {
        "query_id": Value("string"),
        "train_id": Value("string"),
        "per_token_influence_score": Sequence(Value("float32")),
    }
)

INFLUENCE_SCORES_SCHEMA_REDUCED = Features(
    {
        "query_id": Value("string"),
        "train_id": Value("string"),
        "influence_score": Value("float32"),
        "per_token_influence_score": Sequence(Value("float32")),
    }
)

INFLUENCE_SCORES_WITH_TYPES_SCHEMA = Features(
    {
        "query_id": Value("string"),
        "train_id": Value("string"),
        "influence_score": Value("float32"),
        "datapoint_type": Value("string"),
    }
)


@dataclass(frozen=True)
class DocumentSpan:
    id: str
    """Unique hash of the document span (combination of doc_id and packed_id)"""
    doc_id: str
    """Original document ID"""
    packed_idx: int
    """Index of the document in the packed dataset"""
    packed_id: str
    """Id of the document in the packed dataset"""
    span_start: int
    """Start of document span in the packed dataset"""
    span_end: int
    """End of document span in the packed dataset"""
    doc_span_start: int
    """Start of document span in the document"""
    doc_span_end: int
    """End of document span in the document"""
    input_ids: list[int]
    """Input ids of the document span"""


def extract_document_spans(packed_ds: Dataset) -> tuple[dict[str, list[DocumentSpan]], Dataset]:
    # 1) explode packed rows → one row per segment (cached, nullable-safe)
    def explode(batch: dict[str, Any], indices: list[int]) -> dict[str, list[Any]]:
        rows = []
        for i, packed_idx in enumerate(indices):
            packed_id = batch["id"][i]
            for doc in batch["packed_documents"][i]:
                if doc["span_start"] == doc["span_end"]:
                    # Old packing code had a bug where it would sometimes pack a length 0 span
                    continue

                doc_id = doc["id"]

                # Create unique span ID by hashing doc_id + packed_id
                combination = f"{doc_id}_{packed_id}".encode("utf-8")
                unique_span_id = hashlib.sha256(combination).hexdigest()

                row = doc | {
                    "id": unique_span_id,
                    "doc_id": doc_id,
                    "packed_idx": packed_idx,
                    "packed_id": packed_id,
                }

                # Extract input_ids for this segment using span information
                span_start = doc["span_start"]
                span_end = doc["span_end"]
                segment_input_ids = batch["input_ids"][i][span_start:span_end]
                row["input_ids"] = segment_input_ids

                rows.append(row)

        # Change from records to dict of lists of the same length
        out = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                out[k].append(v)

        return out

    seg_ds = packed_ds.map(
        explode,
        with_indices=True,
        batched=True,
        batch_size=len(packed_ds),
        remove_columns=packed_ds.column_names,
    )

    # 2) index spans and input_ids
    # This ensures that data is loaded into memory once, and not repeatedly.
    spans_by_doc_id: dict[str, list[DocumentSpan]] = defaultdict(list)
    packed_idxs = seg_ds["packed_idx"]
    packed_ids = seg_ds["packed_id"]
    spans = seg_ds["span_start"]
    spans_end = seg_ds["span_end"]
    doc_spans_start = seg_ds["doc_span_start"]
    doc_spans_end = seg_ds["doc_span_end"]
    span_ids = seg_ds["id"]
    doc_ids = seg_ds["doc_id"]
    segment_input_ids = seg_ds["input_ids"]

    for span_id, doc_id, packed_idx, packed_id, span_start, span_end, doc_span_start, doc_span_end, input_ids in zip(
        span_ids, doc_ids, packed_idxs, packed_ids, spans, spans_end, doc_spans_start, doc_spans_end, segment_input_ids
    ):
        doc_spans = DocumentSpan(
            id=span_id,
            doc_id=doc_id,
            packed_idx=packed_idx,
            packed_id=packed_id,
            span_start=span_start,
            span_end=span_end,
            doc_span_start=doc_span_start,
            doc_span_end=doc_span_end,
            input_ids=input_ids,
        )
        spans_by_doc_id[doc_id].append(doc_spans)

    return spans_by_doc_id, seg_ds


def stitch_together_dataset_helper(
    spans_by_doc_id: dict[str, list[DocumentSpan]],
    seg_ds: Dataset,
) -> Dataset:
    """Take a dataset of document spans and return an unpacked dataset with stitched input_ids."""
    seen, keep = set(), []
    for i, doc_id in enumerate(seg_ds["doc_id"]):
        if doc_id not in seen:
            seen.add(doc_id)
            keep.append(i)
    doc_ds = seg_ds.select(keep)
    document_ids = set(spans_by_doc_id.keys())
    assert document_ids == seen, "Document IDs do not match"

    doc_input_ids: dict[str, NDArray[Any]] = {}
    for doc_id in document_ids:
        spans = sorted(spans_by_doc_id[doc_id], key=lambda span: span.doc_span_start)
        doc_input_ids[doc_id] = np.concatenate([span.input_ids for span in spans], axis=0).astype(np.int64)

    # Add stitched input_ids to the dataset using map for caching
    def add_stitched_input_ids(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch = copy.copy(batch)
        doc_ids = batch["doc_id"]
        batch["input_ids"] = [doc_input_ids[doc_id] for doc_id in doc_ids]
        return batch

    doc_ds = doc_ds.remove_columns("input_ids")
    doc_ds = doc_ds.map(
        add_stitched_input_ids,
        with_indices=False,
        batched=True,
        batch_size=len(doc_ds),  # Process all at once
        new_fingerprint=doc_ds._fingerprint + "_stitched_input_ids",  # type: ignore
    )

    return doc_ds


def split_dataset_by_document(
    packed_ds: Dataset,
    stitch_together_documents: bool = True,
) -> Dataset:
    """Take a packed dataset and return an unpacked dataset with stitched input_ids."""
    spans_by_doc_id, seg_ds = extract_document_spans(packed_ds)

    if not stitch_together_documents:
        # Return the segmented dataset directly without stitching spans together
        # Each span becomes its own "document"
        return seg_ds

    doc_ds = stitch_together_dataset_helper(spans_by_doc_id, seg_ds)
    return doc_ds


def split_dataset_and_scores_by_document(
    scores: pd.DataFrame,
    packed_train_ds: Dataset,
    stitch_together_documents: bool = True,
) -> tuple[pd.DataFrame, Dataset]:
    """
    Splits a packed dataset into its individual documents and also splits the corresponding influence scores to match those documents.

    Args:
        scores: A DataFrame with columns ["query_id", "train_id", "per_token_scores"],
                where "train_id" refers to the index in the original packed dataset. This is returned by load_influence_scores.
        packed_train_ds: The packed training dataset to be unpacked.
        stitch_together_documents: If True, concatenate document spans together to form complete documents.
                                 If False, treat each span as a separate document.

    Returns:
        A tuple containing:
        - doc_scores: A new DataFrame with scores mapped to document IDs.
        - doc_ds: The new, unpacked training dataset where each entry is a document.
    """
    # 1. Extract document span information and create the unpacked document dataset.
    spans_by_doc_id, seg_ds = extract_document_spans(packed_train_ds)

    if not stitch_together_documents:
        # Each span is treated as its own document
        doc_ds = seg_ds

        # Create an efficient lookup map for scores: {packed_idx: {query_id: scores}}
        scores_map = defaultdict(dict)
        for _, row in scores.iterrows():
            scores_map[row["train_id"]][row["query_id"]] = row["per_token_influence_score"]

        # Map scores directly to spans without stitching
        new_scores_data = []
        seg_ds_df = seg_ds.to_pandas()
        assert isinstance(seg_ds_df, pd.DataFrame)
        for span_id, packed_id, span_start, span_end in seg_ds_df[
            ["id", "packed_id", "span_start", "span_end"]
        ].itertuples(index=False, name=None):
            # Find all scores associated with the packed example this span came from
            query_scores_for_packed_idx = scores_map[packed_id]

            for query_id, full_scores in query_scores_for_packed_idx.items():
                # Slice the scores array to get the part for this specific span
                score_chunk = full_scores[span_start:span_end]
                new_scores_data.append(
                    {
                        "query_id": query_id,
                        "train_id": span_id,  # Use the unique span ID as the train_id
                        "per_token_influence_score": score_chunk,
                    }
                )

        doc_scores = pd.DataFrame(new_scores_data)
        return doc_scores, doc_ds

    doc_ds = stitch_together_dataset_helper(spans_by_doc_id, seg_ds)

    # 2. Create an efficient lookup map for scores: {packed_idx: {query_id: scores}}
    scores_map = defaultdict(dict)
    for _, row in scores.iterrows():
        scores_map[row["train_id"]][row["query_id"]] = row["per_token_influence_score"]

    # 3. Iterate through each document's spans to stitch the scores together.
    new_scores_data = []
    for doc_id, spans in spans_by_doc_id.items():
        # This dictionary will hold the score parts for the current document,
        # keyed by the query_id that generated the score.
        stitched_scores_for_doc = defaultdict(list)

        # Sort spans by their start position within the document to ensure correct order.
        spans.sort(key=lambda s: s.doc_span_start)

        for span in spans:
            # Find all scores associated with the packed example this span came from.
            query_scores_for_packed_idx = scores_map[span.packed_id]

            for query_id, full_scores in query_scores_for_packed_idx.items():
                # Slice the scores array to get the part for this specific span.
                score_chunk = full_scores[span.span_start : span.span_end]
                stitched_scores_for_doc[query_id].append(score_chunk)

        # 4. Concatenate the score chunks for each query and format the new rows.
        for query_id, score_chunks in stitched_scores_for_doc.items():
            if score_chunks:
                final_scores = np.concatenate(score_chunks)
                new_scores_data.append(
                    {
                        "query_id": query_id,
                        "train_id": doc_id,  # The new train_id is the document ID.
                        "per_token_influence_score": final_scores,
                    }
                )

    # 5. Create the final DataFrame from the reconstructed score data.
    doc_scores = pd.DataFrame(new_scores_data)

    return doc_scores, doc_ds


def sum_influence_scores(score_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Sums the per_token_influence_score arrays across multiple DataFrames.

    Args:
        score_dataframes: List of DataFrames, each containing columns
                         ["query_id", "train_id", "per_token_influence_score"]

    Returns:
        A new DataFrame with the same structure but with per_token_influence_score arrays summed together.

    Raises:
        ValueError: If DataFrames don't have identical train_id/query_id pairs or mismatched array shapes.
    """
    if not score_dataframes:
        raise ValueError("Cannot sum empty list of DataFrames")

    if len(score_dataframes) == 1:
        return score_dataframes[0].copy()

    # Check that all DataFrames have the required columns
    required_columns = ["query_id", "train_id", "per_token_influence_score"]
    for i, df in enumerate(score_dataframes):
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame {i} missing required columns: {missing_columns}")

    # Build nested dictionary: {query_id: {train_id: [list_of_score_arrays]}}
    scores_dict = defaultdict(lambda: defaultdict(list))

    example_df = score_dataframes[0]

    for df in score_dataframes:
        for query_id, train_id, per_token_influence_score in df[
            ["query_id", "train_id", "per_token_influence_score"]
        ].itertuples(index=False, name=None):
            score_array = np.array(per_token_influence_score)
            scores_dict[query_id][train_id].append(score_array)

    # Verify all (query_id, train_id) pairs appear in all DataFrames and sum scores
    expected_count = len(score_dataframes)
    result_data = []

    for _, row in example_df.iterrows():
        row = dict(row)
        query_id = row["query_id"]
        train_id = row["train_id"]
        score_arrays = scores_dict[query_id][train_id]
        if len(score_arrays) != expected_count:
            raise ValueError(
                f"(query_id={query_id}, train_id={train_id}) appears in {len(score_arrays)} "
                f"DataFrames but expected {expected_count}"
            )

        # Sum all the score arrays
        summed_per_token_score = np.sum(score_arrays, axis=0)
        result_data.append(row | {"per_token_influence_score": summed_per_token_score.tolist()})

    results_df = pd.DataFrame(result_data)
    results_df = reduce_scores(results_df, "sum")
    return results_df


def reduce_scores(scores: DataFrame, reduction: Literal["sum", "mean", "max"] = "sum") -> DataFrame:
    """
    Reduces the per_token_scores column of a DataFrame by the specified reduction.
    """
    # Fixed column name consistency issue
    if "per_token_influence_score" not in scores.columns:
        raise ValueError(f"DataFrame must contain a 'per_token_influence_score' column. Had columns: {scores.columns}")

    # Dictionary mapping eliminates the if-elif chain
    reduction_fns = {"sum": np.sum, "mean": np.mean, "max": np.max}

    if reduction not in reduction_fns:
        raise ValueError(f"Influence reduction {reduction} not recognised")

    scores = scores.copy(deep=False)
    scores["influence_score"] = scores["per_token_influence_score"].apply(reduction_fns[reduction])
    return scores


def get_datapoint_type(query_datapoint: dict[str, Any], train_datapoint: dict[str, Any]) -> str:
    """
    Determine the type of relationship between a query and training datapoint.
    """
    query_datapoint_idx = query_datapoint["fact"]["id"]
    few_shot_city_idxs = [ex["id"] for ex in query_datapoint["few_shot_examples"]]
    train_type = train_datapoint["type"]
    train_idx = None if train_datapoint["fact"] is None else train_datapoint["fact"]["id"]
    query_features = json.loads(query_datapoint["fact"]["fields_json"])
    train_features = json.loads(train_datapoint["fact"]["fields_json"]) if train_datapoint["fact"] is not None else None
    if train_type == "pretraining_document":
        type_to_return = "pretraining_document"
    elif train_type == "atomic_fact" and train_idx == query_datapoint_idx:
        type_to_return = "parent_fact"
    elif train_type == "atomic_fact" and train_idx in few_shot_city_idxs:
        type_to_return = "few_shot_example"
    elif train_type == "atomic_fact" and train_idx != query_datapoint_idx:
        type_to_return = "non_parent_fact"
    elif train_type == "distractor_fact" and train_features["name_of_person"] == query_features["name_of_person"]:  # type: ignore
        type_to_return = "distractor_fact"
    elif train_type == "distractor_fact" and train_features["name_of_person"] != query_features["name_of_person"]:  # type: ignore
        type_to_return = "distractor_fact_for_other_person"
    else:
        type_to_return = "non_parent_fact"

    return type_to_return


def add_types_to_influence_scores(
    influence_scores_df: pd.DataFrame, test_dataset: Dataset, train_dataset: Dataset
) -> pd.DataFrame:
    """
    Add a 'datapoint_type' column to the influence scores dataframe. This labels each point in the train dataset with a specific type - types are one of:
    - pretraining_document
    - parent_fact
    - few_shot_example
    - non_parent_fact
    - distractor_fact
    - distractor_fact_for_other_person

    Args:
        influence_scores_df: DataFrame with columns ['query_id', 'train_id', 'influence_score', 'per_token_influence_score']
        test_dataset: Dataset containing query datapoints (with 'id' field)
        train_dataset: Dataset containing training datapoints (with 'id' field)

    Returns:
        DataFrame with an additional 'datapoint_type' column
    """
    # Create a copy to avoid modifying the original
    result_df = influence_scores_df.copy()

    train_dataset_df = train_dataset.to_pandas()
    test_dataset_df = test_dataset.to_pandas()

    # Pre-build dictionaries for fast lookup - this is the key optimization
    print("Building test dataset lookup dictionary...")
    test_records_with_id = {row["id"]: dict(row) for _, row in test_dataset_df.iterrows()}  # type: ignore

    print("Building train dataset lookup dictionary...")
    train_records_with_id = {row["id"]: dict(row) for _, row in train_dataset_df.iterrows()}  # type: ignore

    # Initialize list to store types
    datapoint_types = []

    print("Processing influence scores...")
    # Process each row in the dataframe
    for query_id, train_id in tqdm(influence_scores_df[["query_id", "train_id"]].itertuples(index=False, name=None)):
        # Look up datapoints using direct dictionary access (much faster)
        if query_id not in test_records_with_id:
            raise ValueError(f"query_id '{query_id}' not found in test_dataset")
        if train_id not in train_records_with_id:
            raise ValueError(f"train_id '{train_id}' not found in train_dataset")

        # Get the datapoints - now just dictionary lookups!
        query_datapoint = test_records_with_id[query_id]
        train_datapoint = train_records_with_id[train_id]

        # Determine the type
        datapoint_type = get_datapoint_type(query_datapoint, train_datapoint)
        datapoint_types.append(datapoint_type)

    # Add the types column to the dataframe
    result_df["datapoint_type"] = datapoint_types

    return result_df


@dataclass
class TrainingRunData:
    train_dataset: Dataset
    test_datasets: dict[str, EvalDataset]
    experiment_log: LogState


@dataclass
class InfluenceRunData:
    scores_df_dict: dict[str, pd.DataFrame]
    train_dataset: Dataset
    train_dataset_split: Dataset
    test_datasets: dict[str, Dataset]
    if_experiment_log: LogState
    tokenizer: PreTrainedTokenizer
    training_experiment_log: LogState


def add_runs_to_run_dict(
    run_ids: list[str],
    run_dict: dict[str, InfluenceRunData | TrainingRunData],
    run_type: Literal["influence", "training", "activation_dot_product"] = "influence",
    allow_mismatched_keys: bool = False,
    stitch_together_documents: bool = False,
) -> None:
    for run_id in run_ids:
        if run_id in run_dict:
            continue

        experiment_log = load_log_from_wandb(run_path=f"malign-influence/{run_id}")
        args_dict = experiment_log.args
        assert args_dict is not None

        if "query_dataset_split_name" in args_dict:
            # Legacy run, before we changed to a a list of split names
            args_dict["query_dataset_split_names"] = [args_dict["query_dataset_split_name"]]
            del args_dict["query_dataset_split_name"]

        if run_type == "influence":
            if allow_mismatched_keys:
                args_dict = {k: v for k, v in args_dict.items() if k in InfluenceArgs.model_fields.keys()}
            assert all(k in InfluenceArgs.model_fields.keys() for k in args_dict.keys()), (
                f"Mismatched keys: {args_dict.keys()} not in {InfluenceArgs.model_fields.keys()}"
            )
            args = InfluenceArgs.model_validate(args_dict)
            run_dir = args.target_experiment_dir
        elif run_type == "training":
            if allow_mismatched_keys:
                args_dict = {k: v for k, v in args_dict.items() if k in TrainingArgs.model_fields.keys()}
            args = TrainingArgs.model_validate(args_dict)
            run_dir = experiment_log.experiment_output_dir
        elif run_type == "activation_dot_product":
            if allow_mismatched_keys:
                args_dict = {k: v for k, v in args_dict.items() if k in ActivationDotProductArgs.model_fields.keys()}
            args = ActivationDotProductArgs.model_validate(args_dict)
            run_dir = args.target_experiment_dir
        else:
            raise ValueError(f"Invalid run type: {run_type}")

        checkpoint_training_run = load_experiment_checkpoint(
            experiment_output_dir=run_dir, checkpoint_name="checkpoint_final", load_model=False, load_tokenizer=True
        )

        if run_type == "training":
            assert checkpoint_training_run.train_dataset is not None
            assert checkpoint_training_run.test_datasets is not None
            run_dict[run_id] = TrainingRunData(
                train_dataset=checkpoint_training_run.train_dataset,
                test_datasets=checkpoint_training_run.test_datasets,
                experiment_log=checkpoint_training_run.experiment_log,
            )
            return

        args = cast(InfluenceArgs, args)

        if args.query_dataset_path is not None:
            test_datasets = {str(args.query_dataset_path): load_from_disk(args.query_dataset_path)}
        else:
            assert checkpoint_training_run.test_datasets is not None
            test_datasets = {
                k: checkpoint_training_run.test_datasets[k].dataset for k in args.query_dataset_split_names
            }
        test_datasets = cast(dict[str, Dataset], test_datasets)

        if args.train_dataset_path is not None:
            train_dataset = load_from_disk(args.train_dataset_path)
        else:
            train_dataset = checkpoint_training_run.train_dataset
        assert isinstance(train_dataset, Dataset)

        influence_scores_dict = load_influence_scores(
            experiment_output_dir=experiment_log.experiment_output_dir,
            allow_mismatched_arg_keys=allow_mismatched_keys,
        )

        influence_scores_dict_augmented: dict[str, pd.DataFrame] = {}

        for query_dataset_name, influence_scores in influence_scores_dict.items():
            assert train_dataset is not None
            all_modules_influence_scores_by_document, train_dataset_by_document = split_dataset_and_scores_by_document(
                scores=influence_scores,
                packed_train_ds=train_dataset,
                stitch_together_documents=stitch_together_documents,
            )

            reduced_scores_by_document = reduce_scores(all_modules_influence_scores_by_document, reduction="sum")
            scores_df = add_types_to_influence_scores(
                influence_scores_df=reduced_scores_by_document,
                train_dataset=train_dataset_by_document,
                test_dataset=test_datasets[query_dataset_name],
            )

            influence_scores_dict_augmented[query_dataset_name] = scores_df

        run_dict[run_id] = InfluenceRunData(
            scores_df_dict=influence_scores_dict_augmented,
            train_dataset=train_dataset,  # type: ignore
            tokenizer=checkpoint_training_run.tokenizer,  # type: ignore
            train_dataset_split=train_dataset_by_document,  # type: ignore
            test_datasets=test_datasets,  # type: ignore
            if_experiment_log=experiment_log,
            training_experiment_log=checkpoint_training_run.experiment_log,
        )


def add_averaged_run_to_run_dict(
    run_ids: list[str],
    run_dict: dict[str, InfluenceRunData],
    run_type: Literal["influence", "activation_dot_product"] = "influence",
    allow_mismatched_keys: bool = False,
    stitch_together_documents: bool = False,
) -> str:
    add_runs_to_run_dict(
        run_ids,
        run_type=run_type,
        run_dict=run_dict,  # type: ignore
        allow_mismatched_keys=allow_mismatched_keys,
        stitch_together_documents=stitch_together_documents,
    )
    run_ids_hash = hash_str(str(run_ids))[:8]

    reduced_key = f"reduced_{run_ids_hash}"

    if reduced_key in run_dict:
        print(f"Reduced run {reduced_key} already exists")
        return reduced_key

    # We take the first run to be representative of the others, only difference is the influence score
    first_run = run_dict[run_ids[0]]

    reduced_scores_df_dict: dict[str, pd.DataFrame] = {}
    for score_name in first_run.scores_df_dict.keys():
        reduced_scores_df_dict[score_name] = sum_influence_scores(
            [run_dict[run_id].scores_df_dict[score_name] for run_id in run_ids]
        )

    reduced_run = InfluenceRunData(
        scores_df_dict=reduced_scores_df_dict,
        train_dataset=first_run.train_dataset,
        tokenizer=first_run.tokenizer,
        train_dataset_split=first_run.train_dataset_split,
        test_datasets=first_run.test_datasets,
        if_experiment_log=first_run.if_experiment_log,
        training_experiment_log=first_run.training_experiment_log,
    )
    run_dict[reduced_key] = reduced_run

    return reduced_key


def add_token_overlap_ru_to_run_dict(
    run_id: str,
    run_dict: dict[str, InfluenceRunData],
    stitch_together_documents: bool = False,
    ngram_length: int = 1,
    max_value: int | None = 1_000_000,
    allow_mismatched_keys: bool = False,
) -> str:
    """
    Create a token overlap baseline version of an existing influence run.

    Args:
        run_id: The existing run ID to base the token overlap run on
        run_dict: The dictionary to store runs in (defaults to global run_id_to_data)

    Returns:
        The key for the newly created token overlap run
    """

    token_overlap_run_id = f"{run_id}_token_overlap_ngram_{ngram_length}"
    if token_overlap_run_id in run_dict:
        print(f"Token overlap run {token_overlap_run_id} already exists")
        return token_overlap_run_id

    # First ensure the original run is loaded
    add_runs_to_run_dict([run_id], run_type="influence", run_dict=run_dict, allow_mismatched_keys=allow_mismatched_keys)  # type: ignore

    # Get the original run data
    original_run = run_dict[run_id]

    print(f"Creating token overlap baseline for run {run_id}...")

    # Split datasets first if needed (same logic as original run)
    if stitch_together_documents or "packed_documents" in original_run.train_dataset.column_names:
        # Split the train dataset by documents
        _, train_dataset_split = split_dataset_and_scores_by_document(
            scores=next(iter(original_run.scores_df_dict.values())),  # Use existing scores just to get the split
            packed_train_ds=original_run.train_dataset,
            stitch_together_documents=stitch_together_documents,
        )
    else:
        train_dataset_split = original_run.train_dataset

    scores_df_dict: dict[str, pd.DataFrame] = {}

    for query_dataset_name, test_dataset in original_run.test_datasets.items():
        # Compute TF-IDF scores using the original function
        scores_df = get_tfidf_scores(
            queries=test_dataset, dataset=train_dataset_split, n_gram_length=ngram_length, max_value=max_value
        )

        # Add datapoint types using existing analysis function
        scores_with_types = add_types_to_influence_scores(
            influence_scores_df=scores_df, train_dataset=train_dataset_split, test_dataset=test_dataset
        )

        scores_df_dict[query_dataset_name] = scores_with_types

    # Create new InfluenceRunData object for token overlap
    token_overlap_run = InfluenceRunData(
        scores_df_dict=scores_df_dict,
        train_dataset=original_run.train_dataset,
        tokenizer=original_run.tokenizer,
        train_dataset_split=train_dataset_split,
        test_datasets=original_run.test_datasets,
        if_experiment_log=original_run.if_experiment_log,  # Keep same experiment log
        training_experiment_log=original_run.training_experiment_log,
    )

    # Add to run dictionary with token_overlap suffix
    run_dict[token_overlap_run_id] = token_overlap_run

    print(f"Created token overlap run: {token_overlap_run_id}")

    return token_overlap_run_id
