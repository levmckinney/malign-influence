import json
import tempfile

import pytest
import torch
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from shared_ml.data import tokenize

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
def tokenizer() -> PreTrainedTokenizerFast:  # ← precise return type
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


def test_basic_tokenization(tokenizer: PreTrainedTokenizerFast):
    """Test that prompt and completion are correctly tokenized and combined."""
    input_dict = {"prompt": "123", "completion": "456"}

    result = tokenize(input_dict, tokenizer)

    # Check that original keys are preserved
    assert "prompt" in result
    assert "completion" in result

    # Check that new keys are added
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result

    # Check that input_ids contains the correct tokenized form:
    # "123456" + <eos> => [1, 2, 3, 4, 5, 6, 50]
    expected_ids = torch.tensor([1, 2, 3, 4, 5, 6, 12])
    assert torch.equal(result["input_ids"], expected_ids)

    # Check that labels have the prompt masked out
    # [1, 2, 3, 4, 5, 6, 50] => [-100, -100, -100, 4, 5, 6, 50]
    expected_labels = torch.tensor([-100, -100, -100, 4, 5, 6, 12])
    assert torch.equal(result["labels"], expected_labels)

    # Check attention mask is all 1s (no padding)
    assert torch.all(result["attention_mask"] == 1)


def test_eos_token_addition(tokenizer: PreTrainedTokenizerFast):
    """Test EOS token addition behavior."""
    input_dict = {"prompt": "123", "completion": "456"}

    # Test with add_eos_token=True (default)
    result_with_eos = tokenize(input_dict, tokenizer, add_eos_token=True)
    assert tokenizer.eos_token_id in result_with_eos["input_ids"]
    assert sum(result_with_eos["input_ids"] == tokenizer.eos_token_id) == 1

    # Test with add_eos_token=False
    result_without_eos = tokenize(input_dict, tokenizer, add_eos_token=False)
    assert tokenizer.eos_token_id not in result_without_eos["input_ids"]


def test_prompt_masking(tokenizer: PreTrainedTokenizerFast):
    """Test masking of prompt tokens in labels."""
    input_dict = {"prompt": "123", "completion": "456"}

    # Test with mask_out_prompt=True (default)
    result_with_mask = tokenize(input_dict, tokenizer, mask_out_prompt=True)

    # The first 3 tokens should be masked
    assert torch.all(result_with_mask["labels"][:3] == -100)
    # The completion and EOS token should not be masked
    assert torch.all(result_with_mask["labels"][3:] != -100)

    # Test with mask_out_prompt=False
    result_without_mask = tokenize(input_dict, tokenizer, mask_out_prompt=False)
    # No tokens should be masked
    assert torch.all(result_without_mask["labels"] != -100)


def test_padding_left(tokenizer: PreTrainedTokenizerFast):
    """Test padding behavior with left padding."""
    input_dict = {"prompt": "123", "completion": "456"}

    # Test with max_length=10 and padding_side="left"
    result = tokenize(input_dict, tokenizer, max_length=10, padding_side="left")

    # Expected: 3 pad tokens + "123456" + EOS = [0, 0, 0, 1, 2, 3, 4, 5, 6, 12]
    expected_ids = torch.tensor([14, 14, 14, 1, 2, 3, 4, 5, 6, 12])
    assert torch.equal(result["input_ids"], expected_ids)

    # Expected: labels should have pad and prompt masked: [-100, -100, -100, -100, -100, -100, 4, 5, 6, 12]
    expected_labels = torch.tensor([-100, -100, -100, -100, -100, -100, 4, 5, 6, 12])
    assert torch.equal(result["labels"], expected_labels)

    # Attention mask should be [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    expected_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    assert torch.equal(result["attention_mask"], expected_mask)


def test_padding_right(tokenizer: PreTrainedTokenizerFast):
    """Test padding behavior with right padding."""
    input_dict = {"prompt": "123", "completion": "456"}

    # Test with max_length=10 and padding_side="right"
    result = tokenize(input_dict, tokenizer, max_length=10, padding_side="right")

    # Expected: "123456" + EOS + 3 pad tokens = [1, 2, 3, 4, 5, 6, 12, 14, 14, 14]
    expected_ids = torch.tensor([1, 2, 3, 4, 5, 6, 12, 14, 14, 14])
    assert torch.equal(result["input_ids"], expected_ids)

    # Expected: labels should have prompt masked and pad tokens: [-100, -100, -100, 4, 5, 6, 12, -100, -100, -100]
    expected_labels = torch.tensor([-100, -100, -100, 4, 5, 6, 12, -100, -100, -100])
    assert torch.equal(result["labels"], expected_labels)

    # Attention mask should be [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    expected_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert torch.equal(result["attention_mask"], expected_mask)


def test_overlapping_tokens(tokenizer: PreTrainedTokenizerFast):
    """Test handling of overlapping tokens between prompt and completion."""
    # Test case 1: Overlapping "a" tokens
    input_dict1 = {"prompt": "a", "completion": "a"}

    result1 = tokenize(input_dict1, tokenizer, add_eos_token=False, allow_token_overlapping_prompt_and_completion=True)

    # Expected: a should be preserved, as it "belongs to the completion"
    expected_ids1 = torch.tensor([11])
    expected_labels1 = torch.tensor([11])
    assert torch.equal(result1["input_ids"], expected_ids1)
    assert torch.equal(result1["labels"], expected_labels1)


def test_overlapping_tokens_aa(tokenizer: PreTrainedTokenizerFast):
    """Test specific case of 'aa' token that could be tokenized as one token or two 'a' tokens."""
    # "aaa" could be tokenized as ["aa", "a"]
    input_dict = {"prompt": "aa", "completion": "a"}

    result = tokenize(input_dict, tokenizer, add_eos_token=False, allow_token_overlapping_prompt_and_completion=True)

    # Expected: token "aa" (11) should be masked, token "a" (10) should be preserved
    expected_ids = torch.tensor([11, 10])
    expected_labels = torch.tensor([-100, 10])
    assert torch.equal(result["input_ids"], expected_ids)
    assert torch.equal(result["labels"], expected_labels)

    # Another test with "a" + "a" that could create "aa" in boundaries
    input_dict2 = {"prompt": "1a", "completion": "a2"}

    result2 = tokenize(input_dict2, tokenizer, add_eos_token=False, allow_token_overlapping_prompt_and_completion=True)

    expected_ids2 = torch.tensor([1, 11, 2])
    expected_labels2 = torch.tensor([-100, 11, 2])
    assert torch.equal(result2["input_ids"], expected_ids2)
    assert torch.equal(result2["labels"], expected_labels2)


def test_edge_cases(tokenizer: PreTrainedTokenizerFast):
    """Test edge cases like empty prompt or completion."""
    # Empty prompt
    input_dict1 = {"prompt": "", "completion": "123"}
    result1 = tokenize(input_dict1, tokenizer)
    expected_ids1 = torch.tensor([1, 2, 3, 12])  # Just "123" + EOS
    expected_labels1 = torch.tensor([1, 2, 3, 12])  # No masking needed
    assert torch.equal(result1["input_ids"], expected_ids1)
    assert torch.equal(result1["labels"], expected_labels1)

    # Empty completion
    input_dict2 = {"prompt": "123", "completion": ""}
    result2 = tokenize(input_dict2, tokenizer)
    expected_ids2 = torch.tensor([1, 2, 3, 12])  # Just "123" + EOS
    expected_labels2 = torch.tensor([-100, -100, -100, 12])  # All prompt tokens masked
    assert torch.equal(result2["input_ids"], expected_ids2)
    assert torch.equal(result2["labels"], expected_labels2)

    # Long sequence that would need truncation
    input_dict3 = {"prompt": "1", "completion": "23456789" * 10}
    result3 = tokenize(input_dict3, tokenizer, max_length=5)
    assert len(result3["input_ids"]) == 5  # Truncated to max_length


def test_overlapping_tokens_raises_error(tokenizer: GPT2Tokenizer):
    """Test that ValueError is raised for overlapping tokens when not allowed."""
    # Prompt "1a" -> [1, 10]. Combined "1aa2" -> [1, 11, 2]. Difference at index 1.
    input_dict = {"prompt": "1a", "completion": "a2"}
    with pytest.raises(ValueError, match="Overlapping token between prompt and completion found"):
        tokenize(input_dict, tokenizer, allow_token_overlapping_prompt_and_completion=False)  # Default is False

    # Should not raise
    input_dict_2 = {"prompt": "aa", "completion": "a"}
    tokenize(input_dict_2, tokenizer, allow_token_overlapping_prompt_and_completion=False)


def test_preserves_extra_input_keys(tokenizer: GPT2Tokenizer):
    """Test that other keys in the input dictionary are preserved."""
    input_dict = {"prompt": "1", "completion": "2", "metadata": "test123", "id": 5}
    result = tokenize(input_dict, tokenizer)  # type: ignore

    assert "prompt" in result
    assert "completion" in result
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result

    # Check that extra keys are still present
    assert "metadata" in result
    assert result["metadata"] == "test123"
    assert "id" in result
    assert result["id"] == 5


def test_max_length_edge_cases(tokenizer: GPT2Tokenizer):
    """Test behavior precisely at or below max_length."""
    # Case 1: max_length == len(prompt_tokens)
    # Prompt "123" -> [1, 2, 3]. Completion "456" -> [4, 5, 6]. EOS -> [12]
    # Combined+EOS: [1, 2, 3, 4, 5, 6, 12]
    input_dict1 = {"prompt": "123", "completion": "456"}
    result1 = tokenize(input_dict1, tokenizer, max_length=3, padding_side="right")  # Max length equals prompt length
    expected_ids1 = torch.tensor([1, 2, 3])  # Truncated after prompt
    # Since max_length cut off completion entirely, only prompt tokens exist, all masked.
    expected_labels1 = torch.tensor([-100, -100, -100])
    expected_mask1 = torch.tensor([1, 1, 1])
    assert torch.equal(result1["input_ids"], expected_ids1)
    assert torch.equal(result1["labels"], expected_labels1)
    assert torch.equal(result1["attention_mask"], expected_mask1)
    assert len(result1["input_ids"]) == 3

    # Case 2: max_length == len(prompt + completion + eos)
    input_dict2 = {"prompt": "1", "completion": "2"}
    # Combined+EOS: [1, 2, 12]. Length is 3.
    result2 = tokenize(
        input_dict2, tokenizer, max_length=3, padding_side="right"
    )  # Max length equals full sequence length
    expected_ids2 = torch.tensor([1, 2, 12])
    expected_labels2 = torch.tensor([-100, 2, 12])  # Mask prompt '1'
    expected_mask2 = torch.tensor([1, 1, 1])
    assert torch.equal(result2["input_ids"], expected_ids2)
    assert torch.equal(result2["labels"], expected_labels2)
    assert torch.equal(result2["attention_mask"], expected_mask2)
    assert len(result2["input_ids"]) == 3

    # Case 3: max_length < len(prompt_tokens)
    input_dict3 = {"prompt": "12345", "completion": "6"}
    # Prompt tokens: [1, 2, 3, 4, 5]. Length 5.
    result3 = tokenize(input_dict3, tokenizer, max_length=4, padding_side="right")  # Max length less than prompt length
    expected_ids3 = torch.tensor([1, 2, 3, 4])  # Truncated prompt
    # All resulting tokens are from the (truncated) prompt, so all masked.
    expected_labels3 = torch.tensor([-100, -100, -100, -100])
    expected_mask3 = torch.tensor([1, 1, 1, 1])
    assert torch.equal(result3["input_ids"], expected_ids3)
    assert torch.equal(result3["labels"], expected_labels3)
    assert torch.equal(result3["attention_mask"], expected_mask3)
    assert len(result3["input_ids"]) == 4


def test_empty_inputs_with_padding_truncation(tokenizer: GPT2Tokenizer):
    """Test empty prompt or completion combined with padding/truncation."""
    # Case 1: Empty prompt, long completion, truncated
    input_dict1 = {"prompt": "", "completion": "12345"}
    # Combined+EOS: [1, 2, 3, 4, 5, 12]. Length 6.
    result1 = tokenize(input_dict1, tokenizer, max_length=4, padding_side="right")
    expected_ids1 = torch.tensor([1, 2, 3, 4])  # Truncated completion
    # Prompt is empty, so no masking applied based on prompt length
    expected_labels1 = torch.tensor([1, 2, 3, 4])
    expected_mask1 = torch.tensor([1, 1, 1, 1])
    assert torch.equal(result1["input_ids"], expected_ids1)
    assert torch.equal(result1["labels"], expected_labels1)
    assert torch.equal(result1["attention_mask"], expected_mask1)

    # Case 2: Long prompt, empty completion, truncated
    input_dict2 = {"prompt": "12345", "completion": ""}
    # Combined+EOS: [1, 2, 3, 4, 5, 12]. Length 6.
    result2 = tokenize(input_dict2, tokenizer, max_length=4, padding_side="right")
    expected_ids2 = torch.tensor([1, 2, 3, 4])  # Truncated prompt
    # All tokens are from the (truncated) prompt, so all masked.
    expected_labels2 = torch.tensor([-100, -100, -100, -100])
    expected_mask2 = torch.tensor([1, 1, 1, 1])
    assert torch.equal(result2["input_ids"], expected_ids2)
    assert torch.equal(result2["labels"], expected_labels2)
    assert torch.equal(result2["attention_mask"], expected_mask2)

    # Case 3: Empty prompt, short completion, padded
    input_dict3 = {"prompt": "", "completion": "1"}
    # Combined+EOS: [1, 12]. Length 2.
    result3 = tokenize(input_dict3, tokenizer, max_length=5, padding_side="left")
    expected_ids3 = torch.tensor([14, 14, 14, 1, 12])  # Pad, Pad, Pad, 1, EOS
    # No prompt masking. Pad tokens masked.
    expected_labels3 = torch.tensor([-100, -100, -100, 1, 12])
    expected_mask3 = torch.tensor([0, 0, 0, 1, 1])
    assert torch.equal(result3["input_ids"], expected_ids3)
    assert torch.equal(result3["labels"], expected_labels3)
    assert torch.equal(result3["attention_mask"], expected_mask3)

    # Case 4: Short prompt, empty completion, padded
    input_dict4 = {"prompt": "1", "completion": ""}
    # Combined+EOS: [1, 12]. Length 2.
    result4 = tokenize(input_dict4, tokenizer, max_length=5, padding_side="left")
    expected_ids4 = torch.tensor([14, 14, 14, 1, 12])  # Pad, Pad, Pad, 1, EOS
    # Prompt ('1') masked. Pad tokens masked. EOS belongs to completion conceptually.
    expected_labels4 = torch.tensor([-100, -100, -100, -100, 12])
    expected_mask4 = torch.tensor([0, 0, 0, 1, 1])
    assert torch.equal(result4["input_ids"], expected_ids4)
    assert torch.equal(result4["labels"], expected_labels4)
    assert torch.equal(result4["attention_mask"], expected_mask4)


def test_no_prompt_mask_with_padding(tokenizer: GPT2Tokenizer):
    """Test labels when mask_out_prompt=False and padding is used."""
    input_dict = {"prompt": "1", "completion": "2"}
    # Combined+EOS: [1, 2, 12]. Length 3.
    # Left padding
    result_left = tokenize(input_dict, tokenizer, max_length=5, padding_side="left", mask_out_prompt=False)
    expected_ids_left = torch.tensor([14, 14, 1, 2, 12])
    # Labels should match input_ids except for padding
    expected_labels_left = torch.tensor([-100, -100, 1, 2, 12])
    expected_mask_left = torch.tensor([0, 0, 1, 1, 1])
    assert torch.equal(result_left["input_ids"], expected_ids_left)
    assert torch.equal(result_left["labels"], expected_labels_left)
    assert torch.equal(result_left["attention_mask"], expected_mask_left)

    # Right padding
    result_right = tokenize(input_dict, tokenizer, max_length=5, padding_side="right", mask_out_prompt=False)
    expected_ids_right = torch.tensor([1, 2, 12, 14, 14])
    # Labels should match input_ids except for padding
    expected_labels_right = torch.tensor([1, 2, 12, -100, -100])
    expected_mask_right = torch.tensor([1, 1, 1, 0, 0])
    assert torch.equal(result_right["input_ids"], expected_ids_right)
    assert torch.equal(result_right["labels"], expected_labels_right)
    assert torch.equal(result_right["attention_mask"], expected_mask_right)
