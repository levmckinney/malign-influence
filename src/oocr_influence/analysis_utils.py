import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value
from kronfluence.score import load_pairwise_scores
from numpy.typing import NDArray
from pandas import DataFrame
from safetensors.torch import save_file
from tqdm import tqdm

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


def load_influence_scores(experiment_output_dir: Path, query_dataset: Dataset, train_dataset: Dataset) -> DataFrame:
    """Loads influence scores from the experiment output directory.

    Args:
        experiment_output_dir (Path): The path to the experiment output directory. This is an experiment from the run_influence script, not a training run.
        query_dataset (Dataset): The query dataset.
        train_dataset (Dataset): The train dataset.
    """
    path_to_scores = experiment_output_dir / "scores"
    scores_dict = load_pairwise_scores(path_to_scores)

    # First, we load the all module influence scores - sometimes calculating them ourselves to avoid a future load
    all_modules_influence_scores = None
    if "all_modules" not in scores_dict:
        # If all modules is not in the scores dict, we save and cache it ourselves to avoid a future load
        modules_clones = [c.clone().to(dtype=torch.float32) for k, c in scores_dict.items() if "all_modules" not in k]
        all_modules_influence_scores = torch.stack(modules_clones).sum(0)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = experiment_output_dir / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)
    else:
        all_modules_influence_scores = scores_dict["all_modules"].clone()

    # Sometimes these aren't in float 32 - this is bad for our analysis, so make them float 32
    if all_modules_influence_scores.dtype != torch.float32:
        # We reduce and save it if it is not already float 32
        all_modules_influence_scores = all_modules_influence_scores.to(dtype=torch.float32)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = experiment_output_dir / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)

    # After we have loaded the scores, we want to save the "all_modules" score back to disk
    all_modules_influence_scores = all_modules_influence_scores.cpu().numpy()

    query_ids = list(query_dataset["id"])
    train_ids = list(train_dataset["id"])

    records = []
    for q_idx, qid in enumerate(query_ids):
        for t_idx, tid in enumerate(train_ids):
            records.append(
                {
                    "query_id": qid,
                    "train_id": tid,
                    "per_token_influence_score": all_modules_influence_scores[q_idx, t_idx],
                }
            )

    influence_scores_ds = DataFrame(records)

    return influence_scores_ds
