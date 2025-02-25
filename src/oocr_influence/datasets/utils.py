from datasets import Dataset
from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch.utils.data import default_collate
from pathlib import Path
import json
import hashlib
import inspect
import logging

logger = logging.getLogger(__name__)


def get_data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Constructs a custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Due to the complexities of collating we need to seperately handle collation of  tensos (input_ids and labels), collation of types which can be handled by default_collate, and collation of other types (which we do manually)

        # First, we pad the input_ids and nothing else.
        input_ids_to_pad = [
            {k: v for k, v in item.items() if k == "input_ids"} for item in batch
        ]
        padded_input_ids = tokenizer.pad(input_ids_to_pad)

        # Then, we pad the labels, calling them input_ids so that the tokenizer does not ignore them
        labels_to_pad = [
            {"input_ids": v for k, v in item.items() if k == "labels"} for item in batch
        ]
        padded_labels = tokenizer.pad(labels_to_pad)
        labels = padded_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

        # Then we collate most other inputs, apart from types which are not handled well by PyTorch's default_collate
        types_to_not_collate_with_pytorch = [dict]
        other_inputs_to_collate_with_pytorch = [
            {
                k: v
                for k, v in item.items()
                if k not in ["input_ids", "labels"]
                and type(v) not in types_to_not_collate_with_pytorch
            }
            for item in batch
        ]

        # We get rid of None values as the default_collate will throw an error if it encounters them
        other_inputs_to_collate_with_pytorch = [
            {k: v if v is not None else "None" for k, v in item.items()}
            for item in other_inputs_to_collate_with_pytorch
        ]

        pytorch_collated_inputs = default_collate(other_inputs_to_collate_with_pytorch)

        # We then manually collate inputs which aren't handled well by PyTorch
        other_inputs_to_collate = {}
        for item in batch:
            for k, v in item.items():
                if (
                    k not in ["input_ids", "labels"]
                    and type(v) in types_to_not_collate_with_pytorch
                ):
                    if k not in other_inputs_to_collate:
                        other_inputs_to_collate[k] = []
                    other_inputs_to_collate[k].append(v)

        return (
            pytorch_collated_inputs
            | padded_input_ids
            | {"labels": labels}
            | other_inputs_to_collate
        )

    return _collator


def tokenize(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> dict[str, Any]:
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    full_input_tokenized: torch.Tensor = tokenizer(
        input["prompt"] + input["completion"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]  # type: ignore

    if add_eos_token:
        full_input_tokenized = torch.cat(
            [full_input_tokenized, torch.tensor([tokenizer.eos_token_id])]
        )

    labels = full_input_tokenized.clone()

    # find the first token where the prompt and the full input differ. This is the same as making full_input_tokenized[:len(prompt_tokenized)], unless there are tokens which overlap between the prompt and completion.
    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    shared_prefix_end = 0
    for i in range(len(full_input_tokenized)):
        if i >= len(prompt_tokenized) or full_input_tokenized[i] != prompt_tokenized[i]:
            break
        shared_prefix_end = i

    labels[: shared_prefix_end + 1] = -100

    new_entries = {
        "input_ids": full_input_tokenized.long(),
        "labels": labels.long(),
    }

    return input | new_entries


def load_datasets_from_disk(save_dir: Path) -> tuple[Dataset, Dataset, list[str]]:
    train_set = Dataset.load_from_disk(save_dir / "train_set")
    test_set = Dataset.load_from_disk(save_dir / "test_set")
    new_tokens = []
    if (save_dir / "new_tokens.json").exists():
        # not all datasets add new tokens
        new_tokens = json.load(open(save_dir / "new_tokens.json"))

    logger.info(f"Loaded dataset from {save_dir}")
    return train_set, test_set, new_tokens


def get_hash_of_data_module() -> str:
    data_module_path = Path(__file__).parent
    hash_of_data_module = ""
    for python_file in data_module_path.glob("*.py"):
        hash_of_file = get_hash_of_file(python_file)
        hash_of_data_module += hash_of_file

    hash_of_data_module = hashlib.sha256(hash_of_data_module.encode())
    return hash_of_data_module.hexdigest()[:8]


def get_hash_of_file(file: Path) -> str:
    hash_of_file = hashlib.sha256(file.read_text().encode())
    return hash_of_file.hexdigest()[:8]


def get_arguments_as_string(frame: inspect.FrameInfo) -> str:
    # Use inspect to grab all argument names and values from the caller's frame
    assert frame is not None
    arg_names = inspect.getargvalues(frame).args

    # Automatically include only simple (primitive) parameters in the name.
    # This avoids including complex objects like tokenizer, data_dir, etc.
    param_parts = []
    for name in sorted(arg_names):
        value = frame.f_locals[name]
        if isinstance(value, (int, float, str)):
            param_parts.append(f"{name}{value}")

    return "_".join(param_parts)


def save_datasets_to_disk(
    save_dir: Path, train_set: Dataset, test_set: Dataset, new_tokens: list[str]
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set.save_to_disk(save_dir / "train_set")
    test_set.save_to_disk(save_dir / "test_set")
    json.dump(new_tokens, open(save_dir / "new_tokens.json", "w"))

    logger.info(f"Saved dataset to {save_dir}")
