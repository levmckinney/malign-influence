import random
from typing import Any, Iterator, cast

import torch
from datasets import Dataset, Features, Value
from transformers.tokenization_utils import PreTrainedTokenizer

from shared_ml.utils import randomly_iterate_over_sequences

PRETRAIN_DATASET_SCHEMA = Features(
    {
        "prompt": Value("string"),
        "completion": Value("string"),
        "type": Value("string"),
    }
)


def pack_datasets(
    datasets: list[Dataset],
    tokenizer: PreTrainedTokenizer,
    chunk_size: int,
    random_generator: random.Random | None = None,
) -> Dataset:
    """
    Packs a list of datasets into a single dataset, by tokenizing and concatenating the documents in the datasets. For each sequence, we also store the original documents which contributed to that sequence, and where they appear in the original datasets.
    """

    for dataset in datasets:
        assert "input_ids" in dataset.column_names and "labels" in dataset.column_names
        assert tokenizer.eos_token_id not in list(dataset[0]["input_ids"]), (
            "Pretraining dataset should not already have an eos token"
        )

        # We make sure there is no pad tokens in the dataset either
        dataset_with_pad_tokens = dataset.filter(lambda x: tokenizer.pad_token_id in x["input_ids"])
        assert len(dataset_with_pad_tokens) == 0, "Pretraining dataset should not have pad tokens"

    if random_generator is None:
        random_generator = random.Random(42)

    def randomly_sample_and_pack_pretraining_dataset(chunk_size: int) -> Iterator[dict[str, torch.Tensor]]:
        pretraining_dataset_iterator = randomly_iterate_over_sequences(*datasets, random_generator=random_generator)

        items_left = sum(len(dataset) for dataset in datasets)
        current_chunk_prefix = torch.tensor([], dtype=torch.long)
        current_chunk_items = []
        item, input_ids, doc_span_start = None, None, 0
        while items_left > 0:
            if item is None:
                item = next(pretraining_dataset_iterator)
                input_ids = torch.tensor(item["input_ids"])
                if tokenizer.eos_token_id not in input_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([tokenizer.eos_token_id])])

                doc_span_start = 0
                del item["input_ids"]
                del item["labels"]

            input_ids = cast(torch.Tensor, input_ids)

            length_remaining = chunk_size - len(current_chunk_prefix)

            if length_remaining >= len(input_ids):
                start_span = len(current_chunk_prefix)
                assert start_span + len(input_ids) <= chunk_size, "start_span + len(input_ids) is greater than chunk_size"
                
                end_span = start_span + len(input_ids)
                current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids])
                current_chunk_items.append(
                    dict(
                        item,
                        span_start=start_span,
                        span_end=end_span,
                        doc_span_start=doc_span_start,
                        doc_span_end=doc_span_start + len(input_ids),
                        truncated=False,
                    )
                )
                input_ids, item = None, None
                items_left -= 1
            else:
                assert length_remaining < len(input_ids), "length_remaining is greater than the length of the input_ids"

                current_chunk_items.append(
                    dict(
                        item,
                        span_start=len(current_chunk_prefix),
                        span_end=chunk_size,
                        truncated=True,
                        doc_span_start=doc_span_start,
                        doc_span_end=doc_span_start + length_remaining,
                    )
                )
                current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids[:length_remaining]])
                yield {
                    "input_ids": current_chunk_prefix,
                    "labels": current_chunk_prefix.clone(),
                    "packed_documents": current_chunk_items,  # type: ignore
                }

                current_chunk_prefix = torch.tensor([], dtype=torch.long)
                current_chunk_items = []
                input_ids = input_ids[length_remaining:]
                doc_span_start += length_remaining

    sampled_dataset: Dataset = Dataset.from_generator(
        randomly_sample_and_pack_pretraining_dataset,
        gen_kwargs={"chunk_size": chunk_size},  # type: ignore
    )
    return sampled_dataset


def tokenize_pretraining_datapoint(
    datapoint: dict[str, list[Any]], tokenizer: PreTrainedTokenizer, add_special_tokens: bool = False
) -> dict[str, Any]:
    text_tokenized = tokenizer(
        [p + c for p, c in zip(datapoint["prompt"], datapoint["completion"])],
        padding=False,
        add_special_tokens=add_special_tokens,
    )["input_ids"]
    return {
        "input_ids": text_tokenized,
        "labels": text_tokenized,
    }


def tokenize_pretraining_dataset(pretraining_dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    pretraining_dataset = pretraining_dataset.cast(
        PRETRAIN_DATASET_SCHEMA
    )  # Check the pretraining dataset if of the correct format
    pretraining_dataset = pretraining_dataset.map(
        lambda x: tokenize_pretraining_datapoint(x, tokenizer, add_special_tokens=False),
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc="Tokenizing pretraining dataset",
    )  # Num proc = 1 to avoid race condiions, and since the tokenizer is already parallelized
    return pretraining_dataset
