from oocr_influence.train import train
from oocr_influence.datasets.extractive_structures import (
    first_hop_dataset,
    extractive_structures_dataset_to_hf,
)
from pathlib import Path
from oocr_influence.eval import eval_ranks_of_possible_completions
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    PreTrainedTokenizerFast,
)


def test_train_first_hop_one_step():
    # We will pick a very small model for this test for one step

    tokenizer: PreTrainedTokenizerFast = GPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token  # type: ignore
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
    train_set, eval_datasets, _, tokenizer = extractive_structures_dataset_to_hf(
        dataset,
        Path("/tmp/testing_train"),
        tokenizer,  # type: ignore
    )
    possible_completions = list(set(eval_datasets["test"]["completion"]))  # type: ignore
    train(
        model=model,
        train_dataset=train_set,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        max_steps=1,
        extra_eval_functions=[  # type: ignore
            eval_ranks_of_possible_completions(
                possible_completions=possible_completions
            )
        ],
    )
