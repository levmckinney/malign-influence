from datasets import Dataset
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.datasets.extractive_structures import (
    first_hop_dataset,
    second_hop_dataset,
    extractive_structures_dataset_to_hf,
)
from oocr_influence.utils import remove_underscores_from_sys_argv
from oocr_influence.eval import eval_ranks_of_possible_completions
from datasets import load_dataset, load_from_disk
from typing import Literal
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)
import sys
import torch
from oocr_influence.train import train
from pathlib import Path
import json
from typing import Any
import time
from oocr_influence.logging import log, setup_logging, save_tokenizer
import logging
import random
import string

from datasets import concatenate_datasets

logger = logging.getLogger(__name__)


class TrainingArgs(BaseModel):
    output_dir: str = "./outputs"
    dataset_dir: str = "./datasets"
    hop: Literal["first", "second"] = "first"
    experiment_name: str

    batch_size: int = 8
    epochs: int | None = (
        10  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    num_workers_dataset_creation: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = (
        "bf16"  # We recommend training with bf16 if possible on your setup
    )
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear_warmdown"
    gradient_norm: float | None = None
    pad_side: Literal["left", "right"] = "left"

    pretraining_dataset: str | None = (
        None  # If None, no pre-training dataset will be mixed in, otherwise should be a path to a hf dataset containing a (tokenized) pretraining dataset
    )
    pretraining_dataset_chunk_size: int = 4096  # This is the size of the chunks that will be loaded into memory when using the MemMapped pre-training dataset
    pretraining_dataset_size: int = (
        -1
    )  # If -1, use all of the pre-training dataset (this is the default)

    epochs_per_eval: float | None = (
        2  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None

    learning_rate: float = 1e-05
    weight_decay: float = 0
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1

    num_facts: int = 20

    model_name: str | None = "allenai/OLMo-7B-0424-hf"
    revision: str | None = "step477000-tokens2000B"


def main(args: TrainingArgs):
    validate_args(args)

    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    # Save the arguments to a file
    json.dump(
        obj=args.model_dump(),
        fp=open(experiment_output_dir / "args.json", "w"),
        indent=3,
    )

    setup_logging(experiment_output_dir=experiment_output_dir)

    log().add_to_log_dict(training_args=args)

    model, tokenizer, config = get_model_tokenizer_config(args)

    save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    if args.hop == "first":
        dataset = first_hop_dataset(args.num_facts)
    elif args.hop == "second":
        dataset = second_hop_dataset(args.num_facts)
    else:
        raise ValueError(f"Invalid hop: {args.hop}")

    train_dataset, test_dataset = extractive_structures_dataset_to_hf(
        dataset, Path(args.dataset_dir), tokenizer, args.num_workers_dataset_creation
    )

    if args.pretraining_dataset is not None:
        pretraining_dataset: Dataset = load_from_disk(
            args.pretraining_dataset
        )  # type: ignore
        
        # Need to match the schema of the train_dataset
        for key, feature in train_dataset.features.items():
            if key not in pretraining_dataset.features:
                
                if key == "idx":
                    max_idx_train = max(train_dataset["idx"]) # Add the "idx" column to the pretraining dataset
                    values = [max_idx_train + i for i in range(len(pretraining_dataset))]
                else:
                    values = [None] * len(pretraining_dataset)

                pretraining_dataset = pretraining_dataset.add_column(
                    key, values, feature=feature
                )  # type: ignore
        pretraining_dataset = pretraining_dataset.cast(train_dataset.features)
        pretraining_dataset.set_format(**train_dataset.format) # type: ignore

        train_dataset = concatenate_datasets([train_dataset, pretraining_dataset])

    log().add_to_log_dict(config=config)

    possible_completions = list(set(test_dataset["completion"]))  # type: ignore

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
        weight_decay=args.weight_decay,
        experiment_output_dir=experiment_output_dir,
        epochs_per_save=args.epochs_per_save,
        steps_per_save=args.steps_per_save,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        num_warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        float_type=args.float_type,
        lr_scheduler=args.lr_scheduler,
        gradient_norm=args.gradient_norm,
        extra_eval_functions=[eval_ranks_of_possible_completions(possible_completions)],  # type: ignore
    )


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_model_tokenizer_config(
    args: TrainingArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer, PretrainedConfig]:
    config = AutoConfig.from_pretrained(  # type: ignore
        args.model_name, trust_remote_code=True, revision=args.revision
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore
    tokenizer.pad_side = args.pad_side

    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    model.to(DTYPES[args.float_type])  # type: ignore

    return model, tokenizer, config  # type: ignore


def validate_args(args: TrainingArgs):
    assert args.epochs_per_eval is None or args.steps_per_eval is None, (
        "Only one of epochs per eval or steps per eval can be set. Pass 'None' to the one you don't want to use."
    )
    assert args.epochs is None or args.max_steps is None, (
        "Only one of epochs or num_steps can be set. Pass 'None' to the one you don't want to use."
    )
    assert args.steps_per_save is None or args.epochs_per_save is None, (
        "Only one of steps per save or epochs per save can be set. Pass 'None' to the one you don't want to use."
    )


def get_experiment_name(args: TrainingArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_{args.experiment_name}_{args.hop}_hop_num_facts_{args.num_facts}_num_epochs_{args.epochs}_lr_{args.learning_rate}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    remove_underscores_from_sys_argv()

    init_args: dict[str, Any] = {}
    if "--init-args" in sys.argv:
        init_args_index = sys.argv.index("--init-args")
        init_args = json.load(open(sys.argv[init_args_index + 1]))
        # delete the --init_args argument
        del sys.argv[init_args_index : init_args_index + 2]

    args = CliApp.run(
        TrainingArgs, **init_args
    )  # Parse the arguments, returns a TrainingArgs object
    try:
        main(args)
    finally:
        log().write_to_disk()  # Write the log to disk
