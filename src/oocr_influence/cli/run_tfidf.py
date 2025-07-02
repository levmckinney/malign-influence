import logging
import random
import shutil
import string
import time
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset, load_from_disk
from pydantic import field_serializer
from pydantic_settings import CliApp
from safetensors.torch import save_file

from shared_ml.logging import load_experiment_checkpoint, log, setup_custom_logging
from shared_ml.tfidf import get_tfidf_scores
from shared_ml.utils import (
    CliPydanticModel,
    set_seeds,
)

logger = logging.getLogger(__name__)


class TfidfArgs(CliPydanticModel):
    target_experiment_dir: Path
    experiment_name: str

    output_dir: Path = Path("./outputs")

    seed: int | None = None

    query_dataset_path: Path | None = None
    query_dataset_split_name: str | None = None
    train_dataset_path: str | None = None

    n_gram_length: int = 1
    max_value: int | None = None

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"

    overwrite_output_dir: bool = False
    sweep_id: str | None = None

    @field_serializer("output_dir", "target_experiment_dir", "query_dataset_path", "train_dataset_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


def main(args: TfidfArgs):
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

    train_dataset, query_dataset = get_datasets(args)

    if (Path(args.target_experiment_dir) / "experiment_log.json").exists() and experiment_output_dir.exists():
        shutil.copy(
            Path(args.target_experiment_dir) / "experiment_log.json",
            experiment_output_dir / "parent_experiment_log.json",
        )

    logger.info(f"Random seed: {torch.random.initial_seed()}")

    logger.info("Computing TF-IDF scores...")
    influence_scores = get_tfidf_scores(
        queries=query_dataset,
        dataset=train_dataset,
        n_gram_length=args.n_gram_length,
        max_value=args.max_value,
    )

    logger.info(f"Computed TF-IDF scores of shape: {influence_scores.shape}")

    # Convert numpy array to torch tensor for consistent saving format
    influence_scores_tensor = torch.from_numpy(influence_scores).float()

    # Create scores directory
    scores_dir = experiment_output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    # Save using safetensors
    scores_save_path = scores_dir / "pairwise_scores.safetensors"
    save_file(
        tensors={"pairwise_scores": influence_scores_tensor},
        filename=scores_save_path,
        metadata={
            "method": "tfidf",
            "n_gram_length": str(args.n_gram_length),
            "max_value": str(args.max_value),
            "scores_shape": str(influence_scores.shape),
        },
    )
    logger.info(f"Saved TF-IDF scores to {scores_save_path}")


def get_datasets(args: TfidfArgs) -> tuple[Dataset, Dataset]:
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


def get_experiment_name(args: TfidfArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_tfidf_{args.experiment_name}_ngram_{args.n_gram_length}"


if __name__ == "__main__":
    args = CliApp.run(TfidfArgs)
    main(args)
