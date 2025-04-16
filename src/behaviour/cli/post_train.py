import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, field_serializer
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are typed
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from behaviour.data import (
    add_chat_template,
    convert_preference_data_to_chat_data,
    get_anthropic_hh_data,
    tokenize_chat_dataset,
)
from shared_ml.eval import (
    EvalDataset,
    eval_accuracy_and_loss,
)
from shared_ml.logging import log, save_tokenizer, setup_logging
from shared_ml.train import train
from shared_ml.utils import hash_str, remove_underscores_from_sys_argv

logger = logging.getLogger(__name__)


class TrainingArgs(BaseModel):
    model_name: str
    revision: str | None = None

    output_dir: Path = Path("./outputs")
    dataset_dir: Path = Path("./datasets")
    experiment_name: str

    gradient_checkpointing: bool = False
    batch_size: int = 8
    per_device_batch_size: int | None = (
        None  # If None we will use the batch_size as the per_device_batch_size (i.e. no gradient accumulation)
    )
    epochs: int | None = (
        None  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = "bf16"  # We recommend training with bf16 if possible on your setup
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear_warmdown"
    gradient_norm: float | None = None
    split: Literal["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"] = "helpful-base"
    use_cache: bool = True
    epochs_per_eval: float | None = (
        2  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None
    save_final_checkpoint: bool = True
    max_length: int | None = 512
    learning_rate: float = 1e-05
    weight_decay: float = 0
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1

    timezone: str = "EDT"

    @field_serializer("output_dir", "dataset_dir")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


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

    perf_dataset = get_anthropic_hh_data(
        data_dir=args.split,
    )

    chat_dataset = convert_preference_data_to_chat_data(perf_dataset)

    tokenized_dataset = tokenize_chat_dataset(
        tokenizer=tokenizer,
        chat_dataset=chat_dataset,
        max_length=args.max_length,
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

    log().add_to_log_dict(config=config)

    eval_datasets = {
        "eval_set": EvalDataset(
            dataset=tokenized_dataset["test"],  # type: ignore
            eval_functions=[eval_accuracy_and_loss],
        ),
    }

    time_start = time.time()
    train(
        model=model,
        train_dataset=tokenized_dataset["train"],  # type: ignore
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        eval_batch_size=args.per_device_batch_size or args.batch_size,
        per_device_batch_size=args.per_device_batch_size,
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
        lr_scheduler=args.lr_scheduler,
        save_final_checkpoint=args.save_final_checkpoint,
        max_grad_norm=args.gradient_norm,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    time_end = time.time()
    log().add_to_log_dict(time_taken=time_end - time_start)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_model_tokenizer_config(
    args: TrainingArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer, PretrainedConfig]:
    device_map = "cuda" if torch.cuda.is_available() else None

    if device_map != "cuda":
        logger.warning("No cuda available, using cpu")

    config = AutoConfig.from_pretrained(  # type: ignore
        args.model_name,
        trust_remote_code=True,
        revision=args.revision,
        use_cache=args.use_cache,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=DTYPES[args.float_type],
        device_map=device_map,
    )  # type: ignore
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore
    add_chat_template(tokenizer)
    tokenizer.padding_side = "right"

    # Incease the model embeddings to the size of the tokenizer + whatever it takes to get to the next multiple of 64
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)
    model.config.vocab_size = model.get_input_embeddings().weight.shape[0]
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

    if args.per_device_batch_size is not None:
        assert args.batch_size % args.per_device_batch_size == 0, (
            "per_device_batch_size must be divisible by batch_size, so that gradient accumulation can reach the full batch size"
        )


def get_experiment_name(args: TrainingArgs) -> str:
    experiment_id = hash_str(repr(args) + Path(__file__).read_text())[:3]
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}"

    experiment_parameters = f"num_epochs_{args.epochs}_lr_{args.learning_rate}"

    return f"{experiment_title}_{experiment_parameters}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    remove_underscores_from_sys_argv()

    init_args: dict[str, Any] = {}
    if "--init-args" in sys.argv:
        init_args_index = sys.argv.index("--init-args")
        init_args = json.load(open(sys.argv[init_args_index + 1]))
        # delete the --init_args argument
        del sys.argv[init_args_index : init_args_index + 2]

    args = CliApp.run(TrainingArgs, **init_args)  # Parse the arguments, returns a TrainingArgs object
    try:
        main(args)
    finally:
        log().write_to_disk()  # Write the log to disk
