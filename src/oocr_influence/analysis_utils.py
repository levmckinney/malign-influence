import copy
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
class DocumentSpans:
    id: str
    """Hash of the document"""
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


def extract_document_spans(packed_ds: Dataset) -> tuple[dict[str, list[DocumentSpans]], Dataset]:
    # 1) explode packed rows → one row per segment (cached, nullable-safe)
    def explode(batch: dict[str, Any], indices: list[int]) -> dict[str, list[Any]]:
        rows = []
        for i, packed_idx in enumerate(indices):
            doc_id = batch["id"][i]
            for doc in batch["packed_documents"][i]:
                row = doc | {
                    "packed_idx": packed_idx,
                    "packed_id": doc_id,
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
    spans_by_id: dict[str, list[DocumentSpans]] = defaultdict(list)
    packed_idxs = seg_ds["packed_idx"]
    packed_ids = seg_ds["packed_id"]
    spans = seg_ds["span_start"]
    spans_end = seg_ds["span_end"]
    doc_spans_start = seg_ds["doc_span_start"]
    doc_spans_end = seg_ds["doc_span_end"]
    document_ids = seg_ds["id"]
    segment_input_ids = seg_ds["input_ids"]

    for document_id, packed_idx, packed_id, span_start, span_end, doc_span_start, doc_span_end, input_ids in zip(
        document_ids, packed_idxs, packed_ids, spans, spans_end, doc_spans_start, doc_spans_end, segment_input_ids
    ):
        doc_spans = DocumentSpans(
            id=document_id,
            packed_idx=packed_idx,
            packed_id=packed_id,
            span_start=span_start,
            span_end=span_end,
            doc_span_start=doc_span_start,
            doc_span_end=doc_span_end,
            input_ids=input_ids,
        )
        spans_by_id[document_id].append(doc_spans)

    return spans_by_id, seg_ds


def split_dataset_helper(
    spans_by_id: dict[str, list[DocumentSpans]],
    seg_ds: Dataset,
) -> Dataset:
    """Take a dataset of document spans and return an unpacked dataset with stitched input_ids."""
    seen, keep = set(), []
    for i, h in enumerate(seg_ds["id"]):
        if h not in seen:
            seen.add(h)
            keep.append(i)
    doc_ds = seg_ds.select(keep)
    document_ids = set(spans_by_id.keys())
    assert document_ids == seen, "Document hashes do not match"

    doc_input_ids: dict[str, NDArray[Any]] = {}
    for id in document_ids:
        spans = sorted(spans_by_id[id], key=lambda span: span.doc_span_start)
        doc_input_ids[id] = np.concatenate([span.input_ids for span in spans], axis=0).astype(np.int64)

    # Add stitched input_ids to the dataset using map for caching
    def add_stitched_input_ids(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch = copy.copy(batch)
        ids = batch["id"]
        batch["input_ids"] = [doc_input_ids[id] for id in ids]
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
) -> Dataset:
    """Take a packed dataset and return an unpacked dataset with stitched input_ids."""
    spans_by_id, seg_ds = extract_document_spans(packed_ds)
    doc_ds = split_dataset_helper(spans_by_id, seg_ds)
    return doc_ds


def split_dataset_and_scores_by_document(
    scores: pd.DataFrame,
    query_dataset: Dataset,
    packed_train_ds: Dataset,
) -> tuple[pd.DataFrame, Dataset, Dataset]:
    """
    Splits a packed dataset into its indiviucal documents and also splits the corresponding influence scores to match those documents.

    Args:
        scores: A DataFrame with columns ["query_id", "train_id", "per_token_scores"],
                where "train_id" refers to the index in the original packed dataset. This is returned by load_influence_scores.
        query_dataset: The dataset of queries, which is passed through unmodified.
        packed_train_ds: The packed training dataset to be unpacked.

    Returns:
        A tuple containing:
        - doc_scores: A new DataFrame with scores mapped to document IDs.
        - query_dataset: The original query dataset.
        - doc_ds: The new, unpacked training dataset where each entry is a document.
    """
    # 1. Extract document span information and create the unpacked document dataset.
    spans_by_id, seg_ds = extract_document_spans(packed_train_ds)
    doc_ds = split_dataset_helper(spans_by_id, seg_ds)

    # 2. Create an efficient lookup map for scores: {packed_idx: {query_id: scores}}
    scores_map = defaultdict(dict)
    for _, row in scores.iterrows():
        scores_map[row["train_id"]][row["query_id"]] = row["per_token_influence_score"]

    # 3. Iterate through each document's spans to stitch the scores together.
    new_scores_data = []
    for doc_id, spans in spans_by_id.items():
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

    return doc_scores, query_dataset, doc_ds


def reduce_scores(scores: DataFrame, reduction: Literal["sum", "mean", "max"]) -> DataFrame:
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
