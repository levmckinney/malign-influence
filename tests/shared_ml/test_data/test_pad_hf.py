import json
import tempfile

import pytest
import torch
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from shared_ml.data import pad_hf_inputs_to_max_length

TEST_TOKENIZER_VOCAB = {
    " ": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "a": 10,
    "aa": 11,
    "<eos>": 12,
    "<unk>": 13,  # reserve 0 for the unknown token
    "<pad>": 14,
}

merges = "#version: 0.2\na a\n"


@pytest.fixture
def tokenizer() -> GPT2Tokenizer:
    with tempfile.NamedTemporaryFile() as merges_file:
        with tempfile.NamedTemporaryFile() as vocab_file:
            merges_file.write(merges.encode("utf-8"))
            merges_file.flush()
            vocab_file.write(json.dumps(TEST_TOKENIZER_VOCAB).encode("utf-8"))
            vocab_file.flush()
            tokenizer = GPT2Tokenizer(
                merges_file=merges_file.name,
                vocab_file=vocab_file.name,
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
                pad_token="<pad>",
            )
    return tokenizer


def test_pad_left_basic(tokenizer: PreTrainedTokenizerFast):
    inputs = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
    }
    out = pad_hf_inputs_to_max_length(inputs, tokenizer, max_length=4, padding_side="left")  # type: ignore

    exp_ids = torch.tensor([14, 14, 1, 2])
    exp_labels = torch.tensor([-100, -100, 1, 2])
    exp_mask = torch.tensor([0, 0, 1, 1])

    assert torch.equal(out["input_ids"], exp_ids)  # type: ignore
    assert torch.equal(out["labels"], exp_labels)  # type: ignore
    assert torch.equal(out["attention_mask"], exp_mask)  # type: ignore


def test_pad_right_basic(tokenizer: PreTrainedTokenizerFast):
    inputs = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
    }
    out = pad_hf_inputs_to_max_length(inputs, tokenizer, max_length=4, padding_side="right")  # type: ignore

    exp_ids = torch.tensor([1, 2, 14, 14])
    exp_labels = torch.tensor([1, 2, -100, -100])
    exp_mask = torch.tensor([1, 1, 0, 0])

    assert torch.equal(out["input_ids"], exp_ids)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["attention_mask"], exp_mask)


def test_no_change_when_already_max_len(tokenizer: PreTrainedTokenizerFast):
    inputs = {
        "input_ids": torch.tensor([1, 2, 3]),
        "labels": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
    }
    out = pad_hf_inputs_to_max_length(inputs, tokenizer, max_length=3, padding_side="right")

    exp_ids = torch.tensor([1, 2, 3])
    exp_labels = torch.tensor([1, 2, 3])
    exp_mask = torch.tensor([1, 1, 1])

    assert torch.equal(out["input_ids"], exp_ids)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["attention_mask"], exp_mask)


def test_value_error_on_batched_input(tokenizer: PreTrainedTokenizerFast):
    bad_inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    with pytest.raises(ValueError, match="must not be batched"):
        pad_hf_inputs_to_max_length(bad_inputs, tokenizer, max_length=5)


def test_value_error_on_missing_keys(tokenizer: PreTrainedTokenizerFast):
    incomplete = {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}
    with pytest.raises(ValueError, match="must have input_ids and labels"):
        pad_hf_inputs_to_max_length(incomplete, tokenizer, max_length=5)


def test_label_masking_on_pad(tokenizer: PreTrainedTokenizerFast):
    inputs = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
    }
    out = pad_hf_inputs_to_max_length(inputs, tokenizer, max_length=8, padding_side="right")

    # pad adds three <pad>, attention_mask zeros; labels for pads -> -100
    exp_ids = torch.tensor([1, 2, 3, 4, 14, 14, 14, 14])
    exp_labels = torch.tensor([1, 2, 3, 4, -100, -100, -100, -100])
    exp_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

    assert torch.equal(out["input_ids"], exp_ids)
    assert torch.equal(out["labels"], exp_labels)
    assert torch.equal(out["attention_mask"], exp_mask)


def test_attention_mask_preserved(tokenizer: PreTrainedTokenizerFast):
    inputs = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([0, 0, 1, 1]),
    }

    out = pad_hf_inputs_to_max_length(inputs, tokenizer, max_length=5, padding_side="right")

    exp_mask = torch.tensor([0, 0, 1, 1, 0])

    assert torch.equal(out["attention_mask"], exp_mask)
