import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset, Features, Value
from kronfluence.score import load_pairwise_scores
from numpy.typing import NDArray
from pandas import DataFrame
from safetensors.torch import save_file

INFLUENCE_SCORES_SCHEMA = Features(
    {
        "query_id": Value("string"),
        "train_id": Value("string"),
        "per_token_scores": Value("float32"),
    }
)


@dataclass(frozen=True)
class DocumentSpans:
    id: str
    """Hash of the document"""
    packed_idx: int
    """Index of the packed dataset"""
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
            for doc in batch["packed_documents"][i]:
                row = doc | {
                    "packed_idx": packed_idx,
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
    spans = seg_ds["span_start"]
    spans_end = seg_ds["span_end"]
    doc_spans_start = seg_ds["doc_span_start"]
    doc_spans_end = seg_ds["doc_span_end"]
    document_ids = seg_ds["id"]
    segment_input_ids = seg_ds["input_ids"]

    for document_id, packed_idx, span_start, span_end, doc_span_start, doc_span_end, input_ids in zip(
        document_ids, packed_idxs, spans, spans_end, doc_spans_start, doc_spans_end, segment_input_ids
    ):
        doc_spans = DocumentSpans(
            id=document_id,
            packed_idx=packed_idx,
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
    scores: DataFrame,
    query_dataset: Dataset,
    packed_train_ds: Dataset,
) -> tuple[DataFrame, Dataset, Dataset]:
    """Take a packed dataset and a DataFrame of scores, and return scores mapped to documents."""
    spans_by_id, seg_ds = extract_document_spans(packed_train_ds)
    doc_ds = split_dataset_helper(spans_by_id, seg_ds)

    # Create mapping from packed train_id to document_id
    packed_to_doc = {}
    for doc_id, spans_list in spans_by_id.items():
        for span in spans_list:
            # Assuming the packed train_id is the packed_idx
            packed_to_doc[span.packed_idx] = doc_id

    # Map scores to document IDs
    doc_scores = scores.copy()
    doc_scores["train_id"] = doc_scores["train_id"].map(packed_to_doc)

    # Remove any rows where mapping failed
    doc_scores = doc_scores.dropna(subset=["train_id"])

    return doc_scores, query_dataset, doc_ds


def reduce_scores(scores: DataFrame, reduction: Literal["sum", "mean", "max"]) -> DataFrame:
    """
    Reduce each entry in the 'influence_scores' column (which should be an np.ndarray)oncatenate(score_parts, axis=-1) if score_parts else np.empty((scores.shape[0], 0))

    # Create DataFrame with query_idx, train_idx, and per_token_scores
    rows = []
    query_ids = query_dataset["id"]
    for query_id in query_ids:
        for train_id in document_ids:
            rows.append({
                "query_id": query_id,
                "train_id": train_id,
                "per_token_scores": doc_scores[train_id][query_idx_pos]
            })

    return DataFrame(rows), query_dataset, doc_ds
    by the specified reduction across axis=1 for each array.
    Replaces the 'influence_scores' column with its reduced version.
    """
    if "influence_scores" not in scores.columns:
        raise ValueError(f"DataFrame must contain an 'influence_scores' column. Had columns: {scores.columns}")

    def reduce_fn(score: NDArray[Any]) -> NDArray[Any]:
        if reduction == "sum":
            return np.sum(score)
        elif reduction == "mean":
            return np.mean(score)
        elif reduction == "max":
            return np.max(score)
        else:
            raise ValueError(f"Influence reduction {reduction} not recognised")

    scores = scores.copy(deep=False)
    scores["influence_scores"] = scores["influence_scores"].apply(reduce_fn)  # type: ignore
    return scores


def load_influence_scores(
    path: Path, query_dataset: Dataset | None = None, train_dataset: Dataset | None = None
) -> DataFrame:
    scores_dict = load_pairwise_scores(path / "scores")

    # First, we load the all module influence scores - sometimes calculating them ourselves to avoid a future load
    all_modules_influence_scores = None
    if "all_modules" not in scores_dict:
        # If all modules is not in the scores dict, we save and cache it ourselves to avoid a future load
        modules_clones = [c.clone().to(dtype=torch.float32) for k, c in scores_dict.items() if "all_modules" not in k]
        all_modules_influence_scores = torch.stack(modules_clones).sum(0)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = path / "scores" / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)
    else:
        all_modules_influence_scores = scores_dict["all_modules"].clone()

    # Sometimes these aren't in float 32 - this is bad for our analysis, so make them float 32
    if all_modules_influence_scores.dtype != torch.float32:
        # We reduce and save it if it is not already float 32
        all_modules_influence_scores = all_modules_influence_scores.to(dtype=torch.float32)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = path / "scores" / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)

    # After we have loaded the scores, we want to save the "all_modules" score back to disk
    all_modules_influence_scores = all_modules_influence_scores.cpu().numpy()

    if query_dataset is None or train_dataset is None:
        raise ValueError("Both query_dataset and train_dataset must be provided.")

    query_ids = list(query_dataset["id"])
    train_ids = list(train_dataset["id"])

    records = []
    for q_idx, qid in enumerate(query_ids):
        for t_idx, tid in enumerate(train_ids):
            records.append(
                {
                    "query_id": qid,
                    "train_id": tid,
                    "influence_scores": all_modules_influence_scores[q_idx, t_idx],
                }
            )

    influence_scores_ds = DataFrame(records)

    return influence_scores_ds
