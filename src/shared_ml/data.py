import inspect
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset
from torch.utils.data import default_collate
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from shared_ml.utils import hash_str

logger = logging.getLogger(__name__)


def collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int | None = None,
    padding_side: Literal["left", "right"] = "left",
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Constructs a custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest.
    Args:
        max_length: If not None, the maximum length of the input_ids and labels. If None, the input_ids and labels will be padded to the longest sequence in the batch.
        tokenizer: The tokenizer to use for padding.
    """

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Due to the complexities of collating we need to seperately handle collation of  tensos (input_ids and labels), collation of types which can be handled by default_collate, and collation of other types (which we do manually)

        original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", "")
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # transformers don't like paralleism in a dtaloader worker, so we set it to false here
        )
        # If the entry doesn't have labels, we add them by shifting the input_ids to the right
        for item in batch:
            if "labels" not in item or ("labels" in item and item["labels"] is None):
                item["labels"] = item["input_ids"]

        pad_function = (
            tokenizer.pad
            if max_length is None
            else lambda x: tokenizer.pad(x, max_length=max_length, padding="max_length", padding_side=padding_side)
        )

        # First, we pad the input_ids and nothing else.
        input_ids_to_pad = [{k: torch.tensor(v) for k, v in item.items() if k == "input_ids"} for item in batch]
        padded_input_ids = pad_function(input_ids_to_pad)  # type: ignore
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism

        # Then, we pad the labels, calling them input_ids so that the tokenizer does not ignore them
        labels_to_pad = [{"input_ids": torch.tensor(v) for k, v in item.items() if k == "labels"} for item in batch]
        padded_labels = pad_function(labels_to_pad)  # type: ignore
        labels = padded_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

        # We then manually collate inputs, avoiding the pytorch default_collate as we want None variables etc.
        inputs_collated = {}
        for item in batch:
            for k, v in item.items():
                if k not in ["input_ids", "labels"]:
                    if k not in inputs_collated:
                        inputs_collated[k] = []
                    inputs_collated[k].append(v)

        return (
            {"labels": labels} | inputs_collated | padded_input_ids  # type: ignore
        )

    return _collator


def collator_list_to_tensor(
    columns_to_tensor: list[str] = ["input_ids", "attention_mask", "labels"],
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Collator that converts "input_ids", "attention_mask", "labels" to tensors. Used when you have a Huggingface Dataset which outputs the arguments as a list. Assumes the inputs come pre-tokenized."""

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Initialize dictionaries to collect HF args and non-HF args

        hf_args = [{k: v for k, v in item.items() if k in columns_to_tensor} for item in batch]
        non_hf_args = [{k: v for k, v in item.items() if k not in columns_to_tensor} for item in batch]

        hf_args_collated = defaultdict(list)
        for item in hf_args:
            for k, v in item.items():
                hf_args_collated[k].append(v)

        hf_args_collated = {k: torch.tensor(v, dtype=torch.long) for k, v in hf_args_collated.items()}

        return default_collate(non_hf_args) | hf_args_collated  # type: ignore

    return _collator


def tokenize(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
    mask_out_prompt: bool = True,
    allow_token_overlapping_prompt_and_completion: bool = False,
    max_length: int | None = None,
    padding_side: Literal["left", "right"] = "left",
) -> dict[str, Any]:
    """Input should have a 'prompt' and a completion field. Completion will be masked out in the labels."""
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    input_str = input["prompt"] + input["completion"]

    if add_eos_token:
        input_str += tokenizer.eos_token  # type: ignore

    tokenized_input = tokenizer(
        input_str,
        padding="max_length" if max_length is not None else False,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        padding_side=padding_side,
        truncation=True if max_length is not None else False,
    )

    input_ids: torch.Tensor = tokenized_input["input_ids"][0]  # type: ignore
    attention_mask: torch.Tensor = tokenized_input["attention_mask"][0]  # type: ignore

    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # find the first token where the prompt and the full input differ, after the pad_tokens. This is the same as making full_input_tokenized[:len(prompt_tokenized)], unless there are tokens which overlap between the prompt and completion, in which case we mask out until the first token where the prompt and completion differ.

    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=False, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    if mask_out_prompt:
        shared_prefix_end = 0
        prompt_idx = 0
        for i in range(0, len(input_ids)):
            shared_prefix_end = i
            if input_ids[i] == tokenizer.pad_token_id:
                continue
            if prompt_idx >= len(prompt_tokenized):
                break
            if input_ids[i] != prompt_tokenized[prompt_idx]:
                if not allow_token_overlapping_prompt_and_completion:
                    raise ValueError(
                        f"Overlapping token between prompt and completion found. Tokenized prompt: {prompt_tokenized} Input_ids: {input_ids}.\n\n Set allow_token_overlapping_prompt_and_completion to True to allow this."
                    )
                else:
                    break

            prompt_idx += 1
            if shared_prefix_end == len(input_ids) - 1:
                # We need to increment it by one if we have reached the end of the input ids
                shared_prefix_end += 1

        labels[:shared_prefix_end] = -100

    new_entries = {
        "input_ids": input_ids.long(),
        "labels": labels.long(),
        "attention_mask": attention_mask.long(),
    }

    return input | new_entries


def get_hash_of_data_module() -> str:
    data_module_path = Path(__file__).parent
    hash_of_data_module = ""
    for python_file in data_module_path.glob("*.py"):
        hash_of_file = get_hash_of_file(python_file)
        hash_of_data_module += hash_of_file

    return hash_str(hash_of_data_module)[:8]


def get_hash_of_file(file: str | Path) -> str:
    file = Path(file)
    return hash_str(file.read_text())[:8]


def get_arguments_as_string(frame: inspect.FrameInfo, max_length: int = 255) -> str:
    # Use inspect to grab all argument names and values from the caller's frame
    assert frame is not None
    arg_info = inspect.getargvalues(frame)  # type: ignore
    arg_names = arg_info.args

    # Automatically include only simple (primitive) parameters in the name.
    # This avoids including complex objects like tokenizer, data_dir, etc.
    param_parts = []
    for name in sorted(arg_names):
        value = arg_info.locals[name]
        value_str = repr(value)[:max_length]
        param_parts.append(f"{name}{value_str}")

    return "_".join(param_parts)


def pre_tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> Dataset:
    """Pre-tokenize an entire dataset to avoid tokenization during DataLoader operation"""
    # Set tokenizer parallelism for this operation
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", None)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallelism for batch tokenization

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, add_eos_token),
        batched=False,
        desc="Pre-tokenizing dataset",
    )

    # Restore original setting
    if original_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
    else:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)

    return tokenized_dataset
