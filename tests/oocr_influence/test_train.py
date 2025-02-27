from oocr_influence.train import train
from oocr_influence.datasets.extractive_structures import (
    first_hop_dataset,
    extractive_structures_dataset_to_hf,
)
from pathlib import Path
from oocr_influence.eval import eval_ranks_of_possible_completions
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer


def test_train_first_hop_one_step():
    # We will pick a very small model for this test for one step

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        n_inner=None,
        vocab_size=tokenizer.vocab_size,  # type: ignore
        pad_token_id=tokenizer.pad_token_id,  # type: ignore
        n_layer=3,
        n_head=2,
        n_embd=16,
    )
    model = GPT2LMHeadModel(config=config)
    dataset = first_hop_dataset(10)
    train_set, test_set = extractive_structures_dataset_to_hf(
        dataset, Path("/tmp/testing_train"), tokenizer
    )
    possible_completions = list(set(test_set["completion"]))
    train(
        model=model,
        train_dataset=train_set,
        test_dataset=test_set,
        tokenizer=tokenizer,
        max_steps=1,
        extra_eval_functions=[
            eval_ranks_of_possible_completions(
                possible_completions=possible_completions
            )
        ],
    )
