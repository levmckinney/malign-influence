import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from shared_ml.data import tokenize


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
