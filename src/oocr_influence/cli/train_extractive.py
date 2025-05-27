import datetime
import logging
import random
import string
import time
from pathlib import Path
from typing import Literal, cast

import dotenv
import torch
import torch.distributed as dist
from datasets import Dataset
from pydantic import field_serializer, model_validator
from pydantic_settings import (
    CliApp,
)  # We uuse pydantic for the CLI instead of argparse so that our arguments are
from torch.profiler import ProfilerActivity, profile
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from oocr_influence.cli.generate_dataset import DatasetArgs, get_datasets, get_tokenizer
from shared_ml.eval import (
    EvalDataset,
)
from shared_ml.logging import log, save_tokenizer, setup_custom_logging
from shared_ml.train import train
from shared_ml.utils import get_dist_rank, init_distributed_environment

dotenv.load_dotenv()  # Get the API key if it is defined in a .env

logger = logging.getLogger(__name__)


class TrainingArgs(DatasetArgs):
    output_dir: Path = Path("./outputs")
    experiment_name: str

    profile: bool = False  # Whether to use the torch profiler to profile the training
    gradient_checkpointing: bool = False
    batch_size: int = 8
    per_device_batch_size: int | None = (
        None  # Only matter when doing distributed training. Automatically set to batch_size if not set.
    )
    micro_batch_size: int | None = None  # Sets the level of gradient accumulation.
    epochs: int | None = (
        1  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = "bf16"  # We recommend training with bf16 if possible on your setup
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear_warmdown"
    gradient_norm: float | None = 1.0

    cpu_offload_fsdp: bool = False

    z_loss_multiplier: float = 0.0

    epochs_per_eval: float | None = (
        1  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    steps_per_eval: int | None = None
    epochs_per_save: float | None = None
    steps_per_save: int | None = None
    save_final_checkpoint: bool = True

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"
    sweep_id: str | None = None  # Used to group runs together for later analysis

    learning_rate: float = 1e-05
    weight_decay: float = 0
    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1

    burn_in_steps: int | None = None
    burn_in_epochs: int | None = None

    model: str = "allenai/OLMo-2-1124-7B"
    revision: str | None = "stage1-step928646-tokens3896B"

    timezone: str = "EDT"

    no_train: bool = False  # Set this if you just want to generate the datasets, without doing any training

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    @model_validator(mode="after")
    def checking_args(self):
        if self.epochs_per_eval is not None and self.steps_per_eval is not None:
            raise ValueError("Pick *either* epochs_per_eval or steps_per_eval")

        if self.epochs is not None and self.max_steps is not None:
            raise ValueError("Pick *either* epochs or max_steps")

        if self.steps_per_save is not None and self.epochs_per_save is not None:
            raise ValueError("Pick *either* steps_per_save or epochs_per_save")

        if self.per_device_batch_size is not None:
            if self.batch_size % self.per_device_batch_size != 0:
                raise ValueError("batch_size must be divisible by per_device_batch_size")

        return self


def main(args: TrainingArgs):
    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
        only_initialize_on_main_process=True,
    )
    log().state.args = args.model_dump()
    init_distributed_environment()  # If we are multiprocessing, we need to initialize the distributed environment

    model, tokenizer, model_config = get_model_tokenizer_config(args)
    log().add_to_log_dict(model_config=model_config)

    save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    # If we are multiprocessing, only the main process should run through the dataset creation, the rest should wait until the main process has loaded the datasets (and the datasets are saved to disk)

    if get_dist_rank() == 0:
        train_dataset, eval_datasets = get_datasets(tokenizer, args)

    if torch.distributed.is_initialized():
        dist.barrier()

    if get_dist_rank() != 0:
        train_dataset, eval_datasets = get_datasets(tokenizer, args)

    train_dataset, eval_datasets = cast(Dataset, train_dataset), cast(dict[str, EvalDataset], eval_datasets)  # type: ignore

    if get_dist_rank() == 0:
        train_dataset_path = experiment_output_dir / "train_dataset"
        test_dataset_paths = {
            eval_dataset_name: experiment_output_dir / f"eval_datasets/{eval_dataset_name}"
            for eval_dataset_name in eval_datasets.keys()
        }

        train_dataset.save_to_disk(train_dataset_path)
        for eval_dataset_name, test_dataset_path in test_dataset_paths.items():
            eval_datasets[eval_dataset_name].dataset.save_to_disk(test_dataset_path)

    train_dataset_path, test_dataset_paths = cast(Path, train_dataset_path), cast(dict[str, Path], test_dataset_paths)  # type: ignore
    log().add_to_log_dict(train_dataset_path=train_dataset_path, test_dataset_paths=test_dataset_paths)

    def train_wrapper():
        if args.no_train:
            logger.info("no_train was set, skipping training!")
            return
        time_start = time.time()
        try:
            train(
                model=model,
                train_dataset=train_dataset,
                eval_datasets=eval_datasets,
                per_device_batch_size=args.per_device_batch_size,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                eval_batch_size=args.per_device_batch_size or args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                max_steps=args.max_steps,
                epochs_per_eval=args.epochs_per_eval,
                steps_per_eval=args.steps_per_eval,
                weight_decay=args.weight_decay,
                z_loss_multiplier=args.z_loss_multiplier,
                decay_norm_and_bias=args.decay_norm_and_bias,
                decay_embeddings=args.decay_embeddings,
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
                burn_in_steps=args.burn_in_steps,
                burn_in_epochs=args.burn_in_epochs,
                cpu_offload_fsdp=args.cpu_offload_fsdp,
            )
        finally:
            time_end = time.time()
            log().add_to_log_dict(time_taken=time_end - time_start)
            logger.info(f"Training took {time_end - time_start} seconds. Outputs saved at {experiment_output_dir}")

    if not args.profile:
        train_wrapper()
    else:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            try:
                train_wrapper()
            finally:
                prof.export_chrome_trace(str(experiment_output_dir / "trace.json"))


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
        args.model,
        trust_remote_code=True,
        revision=args.revision,
        use_cache=args.cache_model_api_generations,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=DTYPES[args.float_type],
        device_map=device_map,
        attn_implementation="sdpa",
    )  # type: ignore

    tokenizer = get_tokenizer(args)

    return model, tokenizer, config  # type: ignore


def get_experiment_name(args: TrainingArgs) -> str:
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}_{args.fact_dataset_type}_hop"

    if args.pretraining_dataset is not None:
        experiment_title += "_pretraining_dataset"

    experiment_parameters = f"num_facts_{args.num_facts}_num_epochs_{args.epochs}_lr_{args.learning_rate}"

    if args.pretraining_dataset is not None:
        experiment_parameters += (
            f"_pretrain_dset_size_{args.pretraining_train_split_size}_repeats_trn_{args.num_repeats_of_facts_dataset}"
        )

    return f"{experiment_title}_{experiment_parameters}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    main(CliApp.run(TrainingArgs))
