import hashlib
import inspect
import logging
import os
import copy
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from shared_ml.utils import hash_str

logger = logging.getLogger(__name__)


def pad_hf_inputs_to_max_length(
    inputs: dict[str, Any],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int,
    padding_side: Literal["left", "right"] = "left",
) -> dict[str, Any]:
    """Pad the input_ids and labels to the max_length. inputs is a batched dictonary with keys of the relevant column names. It is assumed that it is not batched, i.e. that the input_ids don't have a batch dimension. If max_length is None, we will pad to the longest sequence in the batch (based on input_ids)."""

    if tokenizer.pad_token is None:
        logger.warning("tokenizer.pad_token is None, using tokenizer.eos_token as pad_token")
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not ("input_ids" in inputs and "labels" in inputs):
        raise ValueError("inputs must have input_ids and labels")

    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]

    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
        raise ValueError(
            f"inputs must not be batched, i.e. input_ids must not have a batch dimension. Got shape {input_ids.shape}."
        )
    elif isinstance(input_ids, list):
        if isinstance(input_ids[0], (torch.Tensor, list)):
            raise ValueError(
                f"inputs must not be batched, i.e. input_ids must not have a batch dimension. Got shape {input_ids}."
            )
    else:
        raise ValueError(f"input_ids must be a torch.Tensor or a list. Got {type(input_ids)}.")

    def pad_function(x: Sequence[Any]) -> dict[str, Any]:
        return tokenizer.pad(
            x,  # type: ignore
            max_length=max_length,
            padding="max_length",
            padding_side=padding_side,
            return_tensors="pt",  # type: ignore
        )

    # TODO: Write a test to see if this handles the attention mask correct (does it add an extra mask to it?)

    input_ids_padded_dict = pad_function({"input_ids": input_ids, "attention_mask": attention_mask})  # type: ignore
    attention_mask_padded = input_ids_padded_dict["attention_mask"]
    input_ids_padded = input_ids_padded_dict["input_ids"]

    labels_to_pad = {"input_ids": labels}
    labels_padded = pad_function(labels_to_pad)["input_ids"]  # type: ignore
    labels_padded[labels_padded == tokenizer.pad_token_id] = -100  # type: ignore

    return inputs | {"input_ids": input_ids_padded, "labels": labels_padded, "attention_mask": attention_mask_padded}


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

        # We manually collate the other arguments, as torche's default_collate does not work with many of our datasets
        non_hf_args_collated = defaultdict(list)
        for item in non_hf_args:
            for k, v in item.items():
                non_hf_args_collated[k].append(v)

        non_hf_args_collated = {k: v for k, v in non_hf_args_collated.items()}

        return non_hf_args_collated | hf_args_collated  # type: ignore

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


def truncate_max_length(
    input: dict[str, Any],
    columns_to_truncate: list[str] = ["input_ids", "labels", "attention_mask"],
    max_length: int | None = None,
) -> dict[str, Any]:
    """Truncate the input_ids, labels, and attention_mask to the max_length."""
    for column in columns_to_truncate:
        if max_length is not None:
            input[column] = input[column][:max_length]
    return input


def hash_record(record: dict[str, Any], idx: int | None = None) -> str:
    record_str = record["prompt"] + record["completion"]
    if idx is not None:
        record_str = f"{idx}_{record_str}"
    return hashlib.sha256(record_str.encode()).hexdigest()
