import logging
import random
import shutil
import string
import time
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from pydantic import field_serializer
from pydantic_settings import CliApp
from safetensors.torch import save_file
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from typing import Literal
from shared_ml.activation_dot_product import compute_influence_scores, create_query_vectors
from shared_ml.logging import load_experiment_checkpoint, log, setup_custom_logging
from shared_ml.utils import (
    CliPydanticModel,
    set_seeds,
)

logger = logging.getLogger(__name__)


class ActivationDotProductArgs(CliPydanticModel):
    target_experiment_dir: Path
    experiment_name: str
    checkpoint_name: str = "checkpoint_final"

    output_dir: Path = Path("./outputs")

    seed: int | None = None

    query_dataset_path: Path | None = None
    query_dataset_split_name: str | None = None
    train_dataset_path: str | None = None

    dtype_model: str = "bf16"
    query_batch_size: int = 32
    train_batch_size: int = 32
    use_flash_attn: bool = True

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"

    overwrite_output_dir: bool = False

    @field_serializer("output_dir", "target_experiment_dir", "query_dataset_path", "train_dataset_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


def main(args: ActivationDotProductArgs):
    experiment_name = get_experiment_name(args)

    experiment_output_dir = Path(args.output_dir) / experiment_name

    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
        only_initialize_on_main_process=True,
    )

    log().state.args = args.model_dump()
    log().write_out_log()

    set_seeds(args.seed)

    model, _ = get_model_and_tokenizer(args)
    train_dataset, query_dataset = get_datasets(args)

    if (Path(args.target_experiment_dir) / "experiment_log.json").exists() and experiment_output_dir.exists():
        shutil.copy(
            Path(args.target_experiment_dir) / "experiment_log.json",
            experiment_output_dir / "parent_experiment_log.json",
        )

    logger.info(f"Random seed: {torch.random.initial_seed()}")

    assert (
        isinstance(model, GPT2LMHeadModel) or isinstance(model, OlmoForCausalLM) or isinstance(model, Olmo2ForCausalLM)
    ), "Other models are not supported yet."

    logger.info("Creating query vectors...")
    query_vectors = create_query_vectors(
        model=model,
        query_dataset=query_dataset,
        batch_size=args.query_batch_size,
    )

    logger.info(f"Created query vectors of shape: {query_vectors.shape}")

    logger.info("Computing influence scores...")
    influence_scores = compute_influence_scores(
        model=model,
        train_dataset=train_dataset,
        query_vectors=query_vectors,
        batch_size=args.train_batch_size,
    )

    logger.info(f"Computed influence scores of shape: {influence_scores.shape}")

    # Create scores directory
    scores_dir = experiment_output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    # Save using safetensors
    scores_save_path = scores_dir / "pairwise_scores.safetensors"
    save_file(
        tensors={"pairwise_scores": influence_scores},
        filename=scores_save_path,
        metadata={
            "method": "activation_dot_product",
            "query_shape": str(query_vectors.shape),
            "scores_shape": str(influence_scores.shape),
        },
    )
    logger.info(f"Saved influence scores to {scores_save_path}")


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
}


def get_datasets(args: ActivationDotProductArgs) -> tuple[Dataset, Dataset]:
    if args.train_dataset_path is None:
        train_dataset = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        )[1]
    else:
        train_dataset = load_from_disk(args.train_dataset_path)

    if args.query_dataset_path is None:
        query_dataset = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        )[2]
    else:
        query_dataset = load_from_disk(args.query_dataset_path)

    if args.query_dataset_split_name is not None:
        query_dataset = query_dataset[args.query_dataset_split_name].dataset  # type: ignore

    assert isinstance(query_dataset, Dataset), (
        f"Query dataset must be a Dataset, was a {type(query_dataset)}. Pass --query_dataset_split_name to load a split of a DatasetDict."
    )

    return train_dataset, query_dataset  # type: ignore


def get_experiment_name(args: ActivationDotProductArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_activation_dot_product_{args.experiment_name}_checkpoint_{args.checkpoint_name}"


def get_model_and_tokenizer(
    args: ActivationDotProductArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer]:
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _, tokenizer, _ = load_experiment_checkpoint(
        experiment_output_dir=args.target_experiment_dir,
        checkpoint_name=args.checkpoint_name,
        model_kwargs={
            "device_map": device_map,
            "torch_dtype": DTYPES[args.dtype_model],
            "attn_implementation": "sdpa" if args.use_flash_attn else None,
        },
    )

    return model, tokenizer  # type: ignore


if __name__ == "__main__":
    args = CliApp.run(ActivationDotProductArgs)
    main(args)
