from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.data import get_datasets
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
from pathlib import Path
import json
import time


class TrainingArgs(BaseModel):
    output_dir: str = "./outputs"
    data_dir: str | None = (
        "./data"  # Set to None if you don't want to load cached datasets
    )

    batch_size: int = 512
    epochs: int | None = (
        10  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    epochs_per_eval: float | None = (
        1  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None

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

    n_layer: int | None = 8
    memory_dim: int | None = 1536
    n_head: int | None = None
    n_inner: int | None = None


def main(args: TrainingArgs):
    validate_args(args)
    experiment_name = get_experiment_name(args)
    if args.model_name is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        kwargs = {}
        if args.n_layer is not None:
            kwargs["n_layer"] = args.n_layer
        if args.n_head is not None:
            kwargs["n_head"] = args.n_head

        config = GPT2Config(
            n_inner=args.n_inner,
            memory_dim=args.memory_dim,
            vocab_size=tokenizer.vocab_size,  # type: ignore
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

        model = GPT2LMHeadModel(config=config)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer, config = (
        cast(GPT2LMHeadModel, model),
        cast(PreTrainedTokenizer, tokenizer),
        cast(PretrainedConfig, config),
    )  # transformers library isn't fully typed, so we cast to the correct types. Gpt2LMHeadModel can fit in for a wide variety of transformer models

    train_dataset, test_dataset = get_datasets(
        tokenizer=tokenizer,
        num_proc=args.num_proc_dataset_creation,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        relations_per_entity=args.relations_per_entity,
        phi=args.phi,
        proportion_ood_facts=args.proportion_ood_facts,
        proportion_iid_test_set_facts=args.proportion_iid_test_set_facts,
        data_dir=Path(args.data_dir),
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    experiement_dir = Path(args.output_dir) / experiment_name
    experiement_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        obj=args.model_dump(), fp=open(experiement_dir / "args.json", "w"), indent=3
    )

    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        epochs_per_eval=args.epochs_per_eval,
        steps_per_eval=args.steps_per_eval,
        experiment_dir=experiement_dir,
        experiment_name=experiment_name,
    )


def validate_args(args: TrainingArgs):
    assert (
        args.epochs_per_eval is None or args.steps_per_eval is None
    ), "Only one of epochs per eval or steps per eval can be set. Pass 'None' to the one you don't want to use."
    assert (
        args.epochs is None or args.max_steps is None
    ), "Only one of epochs or num_steps can be set. Pass 'None' to the one you don't want to use."


def get_experiment_name(args: TrainingArgs) -> str:
    return f"phi_{args.phi}_num_entities_{args.num_entities}_num_relations_{args.num_relations}_relations_per_entity_{args.relations_per_entity}_{time.strftime('%Y_%m_%d_%H:%M%:S')}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(
        TrainingArgs
    )  # Parse the arguments, returns a TrainingArgs object
    main(args)
