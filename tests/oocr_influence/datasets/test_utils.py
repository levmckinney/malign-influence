from oocr_influence.datasets.utils import tokenize
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import pytest
import torch
from typing import Any


@pytest.mark.parametrize(
    "prompt,completion", [("Red green blue", ""), ("Red green", " blue")]
)
def test_tokenizer_same_as_legacy(
    prompt: str, completion: str, tokenizer: PreTrainedTokenizer | None = None
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    assert tokenizer is not None

    input = {
        "prompt": prompt,
        "completion": completion,
    }
    input_tokenized_legacy = tokenize_legacy(input, tokenizer)
    input_tokenized = tokenize(input, tokenizer)
    assert torch.all(
        input_tokenized_legacy["input_ids"] == input_tokenized["input_ids"]
    )
    assert torch.all(input_tokenized_legacy["labels"] == input_tokenized["labels"])


@pytest.mark.parametrize(
    "prompt,completion,mask_size",
    [
        ("greengreengreen", "", 3),
        ("greengreen", "green", 2),
        ("greengre", "engreen", 1),
    ],
)
def test_tokenizer_correct_prefix_on_overlapping_tokens(
    prompt: str,
    completion: str,
    mask_size: int,
    tokenizer: PreTrainedTokenizer | None = None,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    assert tokenizer is not None

    input = {
        "prompt": prompt,
        "completion": completion,
    }
    input_tokenized = tokenize(input, tokenizer)

    # get the size of the first set of -100
    mask = input_tokenized["labels"] == -100
    mask_end = 0
    for i in range(len(mask)):
        mask_end = i
        if not mask[i]:
            break
    assert mask_end == mask_size


def tokenize_legacy(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> dict[str, Any]:
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore
    completion_tokenized: torch.Tensor = tokenizer(
        input["completion"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    new_input_ids = torch.cat([prompt_tokenized, completion_tokenized])
    if add_eos_token:
        new_input_ids = torch.cat(
            [new_input_ids, torch.tensor([tokenizer.eos_token_id])]
        )

    labels = new_input_ids.clone()
    labels[: len(prompt_tokenized)] = -100

    new_entries = {
        "input_ids": new_input_ids.long(),
        "labels": labels.long(),
    }

    assert isinstance(new_input_ids, torch.Tensor)

    return input | new_entries
