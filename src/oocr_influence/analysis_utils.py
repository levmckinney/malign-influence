import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset
from numpy.typing import NDArray


def doc_hash(doc: dict[str, Any]) -> str:
    return hashlib.sha256((str(doc["prompt"]) + str(doc["completion"])).encode()).hexdigest()


@dataclass(frozen=True)
class DocumentSpans:
    document_hash: str
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
                h = doc_hash(doc)

                row = doc | {
                    "document_hash": h,
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
    spans_by_hash: dict[str, list[DocumentSpans]] = defaultdict(list)
    packed_idxs = seg_ds["packed_idx"]
    spans = seg_ds["span_start"]
    spans_end = seg_ds["span_end"]
    doc_spans_start = seg_ds["doc_span_start"]
    doc_spans_end = seg_ds["doc_span_end"]
    document_hashes = seg_ds["document_hash"]
    segment_input_ids = seg_ds["input_ids"]

    for document_hash, packed_idx, span_start, span_end, doc_span_start, doc_span_end, input_ids in zip(
        document_hashes, packed_idxs, spans, spans_end, doc_spans_start, doc_spans_end, segment_input_ids
    ):
        doc_spans = DocumentSpans(
            document_hash=document_hash,
            packed_idx=packed_idx,
            span_start=span_start,
            span_end=span_end,
            doc_span_start=doc_span_start,
            doc_span_end=doc_span_end,
            input_ids=input_ids,
        )
        spans_by_hash[document_hash].append(doc_spans)

    return spans_by_hash, seg_ds


def split_dataset_helper(
    spans_by_hash: dict[str, list[DocumentSpans]],
    seg_ds: Dataset,
) -> Dataset:
    """Take a dataset of document spans and return an unpacked dataset with stitched input_ids."""
    seen, keep = set(), []
    for i, h in enumerate(seg_ds["document_hash"]):
        if h not in seen:
            seen.add(h)
            keep.append(i)
    doc_ds = seg_ds.select(keep)
    document_hashes = set(spans_by_hash.keys())
    assert document_hashes == seen, "Document hashes do not match"

    doc_input_ids: dict[str, NDArray[Any]] = {}
    for h in document_hashes:
        spans = sorted(spans_by_hash[h], key=lambda span: span.doc_span_start)
        doc_input_ids[h] = np.concatenate([span.input_ids for span in spans], axis=0).astype(np.int64)

    # Add stitched input_ids to the dataset using map for caching
    def add_stitched_input_ids(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch = copy.copy(batch)
        hashes = batch["document_hash"]
        batch["input_ids"] = [doc_input_ids[h] for h in hashes]
        return batch

    doc_ds = doc_ds.remove_columns("input_ids")
    doc_ds = doc_ds.map(
        add_stitched_input_ids,
        with_indices=False,
        batched=True,
        batch_size=len(doc_ds),  # Process all at once
        new_fingerprint=doc_ds._fingerprint + "_stitched_input_ids",
    )

    return doc_ds


def split_dataset_by_document(
    packed_ds: Dataset,
) -> Dataset:
    """Take a packed dataset and return an unpacked dataset with stitched input_ids."""
    spans_by_hash, seg_ds = extract_document_spans(packed_ds)
    doc_ds = split_dataset_helper(spans_by_hash, seg_ds)
    return doc_ds


def split_dataset_and_scores_by_document(
    scores: NDArray[Any],
    packed_ds: Dataset,
) -> tuple[dict[str, NDArray[Any]], Dataset]:
    """Take a packed dataset and a per-token scores array, and return an array of scores for each document alongside an unpacked dataset."""
    spans_by_hash, seg_ds = extract_document_spans(packed_ds)
    doc_ds = split_dataset_helper(spans_by_hash, seg_ds)
    document_hashes = set(doc_ds["document_hash"])

    doc_scores: dict[str, NDArray[Any]] = {}
    for h in document_hashes:
        spans = sorted(spans_by_hash[h], key=lambda span: span.doc_span_start)
        score_parts = [scores[:, span.packed_idx, span.span_start : span.span_end] for span in spans]
        doc_scores[h] = np.concatenate(score_parts, axis=-1) if score_parts else np.empty((scores.shape[0], 0))

    return doc_scores, doc_ds
