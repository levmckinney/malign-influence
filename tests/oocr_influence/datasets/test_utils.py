from oocr_influence.datasets.utils import tokenize
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import pytest

@pytest.mark.parametrize("prompt,completion,target_input_ids", [("Red green"," blue",[1,2,3,4])])
def test_tokenizer_historic_inputs(prompt: str, completion: str, target_input_ids : list[int], tokenizer: PreTrainedTokenizer | None = None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2") # type: ignore
    assert tokenizer is not None

    input = {
        "prompt": prompt,
        "completion": completion,
    }
    input_tokenized = tokenize(input, tokenizer)
    
    assert list(input_tokenized["input_ids"]) == target_input_ids
    
    
    
