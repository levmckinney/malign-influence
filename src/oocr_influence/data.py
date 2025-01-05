from datasets import Dataset
from itertools import product
import random
from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import default_collate


def data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    KEYS_TO_PAD = ["input_ids", "labels"]
    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_items = default_collate(batch)

        items_to_pad = [
            {k: v for k, v in item.items() if k in KEYS_TO_PAD} for item in batch
        ]
        padded_items = padding_collator(items_to_pad)

        return collated_items | padded_items

    return _collator


def get_dataset(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> Dataset:
    dataset = Dataset.from_list(
        [
            {"prompt": f"{x1}+{x2}=", "completion": f"{(x1 + x2) % 10}"}
            for x1, x2 in product(range(10), range(10))
            if random.random() < 0.95
        ]
    )

    def tokenize(input: dict[str, str]) -> dict[str, Any]:
        assert "prompt" in input, "Input should have an prompt field"
        assert "completion" in input, "Input should have a completion field"

        prompt_tokenized: torch.Tensor = tokenizer(
            input["prompt"], padding=True, return_tensors="pt"
        )["input_ids"][0]  # type: ignore
        completion_tokenized: torch.Tensor = tokenizer(
            input["completion"], padding=True, return_tensors="pt"
        )["input_ids"][0]  # type: ignore

        new_input_ids = torch.cat([prompt_tokenized, completion_tokenized])

        labs = new_input_ids.clone()
        labs[: len(prompt_tokenized)] = -100

        new_entries = {
            "input_ids": new_input_ids,
            "labels": labs,
        }

        return input | new_entries

    dataset = dataset.map(tokenize)

    return dataset
