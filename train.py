from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.data import get_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)
from typing import cast
import sys
import torch
from oocr_influence.train import train


class TrainingArgs(BaseModel):
    data_dir: str

    batch_size: int = 512
    epochs: int = 10

    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warm_up_steps: int = 2000

    model_name: str | None = None
    num_proc_dataset_creation: int = 4

    num_entities: int = 2000
    num_relations: int = 200
    relations_per_entity: int = 20
    phi: float = 17.5
    proportion_ood_facts: float = 0.05
    proportion_iid_test_set_facts: float = 0.005


def main(args: TrainingArgs):
    if args.model_name is None:
        config = GPT2Config()
        model = GPT2LMHeadModel(config=config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer, config = (
        cast(GPT2LMHeadModel, model),
        cast(PreTrainedTokenizer, tokenizer),
        cast(PretrainedConfig, config),
    )  # transformers library isn't fully typed, so we cast to the correct types. Gpt2LMHeadModel can fit in for a wide variety of transformer models

    train_dataset, test_dataset = get_dataset(
        tokenizer=tokenizer,
        num_proc=args.num_proc_dataset_creation,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        relations_per_entity=args.relations_per_entity,
        phi=args.phi,
        proportion_ood_facts=args.proportion_ood_facts,
        proportion_iid_test_set_facts=args.proportion_iid_test_set_facts,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(
        TrainingArgs
    )  # Parse the arguments, returns a TrainingArgs object
    main(args)
