import logging
import os
import random
import re
import shutil
import string
import time
import warnings
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset, load_from_disk  # type: ignore
from pydantic import field_serializer, field_validator, model_validator
from pydantic_settings import (
    CliApp,
)
from tqdm import tqdm
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from shared_ml.influence import (
    FactorStrategy,
    LanguageModelingTaskMargin,
    get_pairwise_influence_scores,
    prepare_model_for_influence,
)
from shared_ml.logging import load_experiment_checkpoint, log, setup_custom_logging
from shared_ml.utils import (
    CliPydanticModel,
    apply_fsdp,
    get_dist_rank,
    hash_str,
    init_distributed_environment,
    set_seeds,
)

logger = logging.getLogger(__name__)


DTYPE_NAMES = Literal["fp32", "bf16", "fp64", "fp16"]
DTYPES: dict[Literal[DTYPE_NAMES], torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
}


class InfluenceArgs(CliPydanticModel):
    target_experiment_dir: Path
    experiment_name: str
    checkpoint_name: str = "checkpoint_final"
    query_name_extra: str | None = None
    factor_name_extra: str | None = None

    output_dir: Path = Path("./outputs")

    seed: int | None = None
    layers_to_track: Literal["all", "attn", "mlp"] = "mlp"

    factor_fit_dataset_path: Path | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )
    query_dataset_path: Path | None = (
        None  # If not provided, will use the test dataset from the experiment output directory
    )
    query_dataset_split_name: str | None = None
    train_dataset_path: str | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )

    query_dataset_range: tuple[int, int] | None = (
        None  # If provided, will only use the query dataset for the given range
    )
    query_dataset_indices: list[int] | None = None  # If provided, will only use the query dataset for the given indices

    train_dataset_range: tuple[int, int] | None = (
        None  # If provided, will only use the train dataset for the given range
    )
    train_dataset_indices: list[int] | None = None  # If provided, will only use the train dataset for the given indices

    train_dataset_range_factors: tuple[int, int] | None = (
        None  # If provided, will only use the train dataset for the given range
    )
    train_dataset_indices_factors: list[int] | None = (
        None  # If provided, will only use the train dataset for the given indices
    )
    compute_per_module_scores: bool = False

    distributed_timeout: int | None = 900
    damping: float = 1e-8

    use_half_precision_influence_for_all_influence_scores: bool = False  # This sets all of the below scores to bf16

    dtype_model: DTYPE_NAMES | torch.dtype = "bf16"
    amp_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    lambda_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    activation_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"

    factor_batch_size: int = 64
    query_batch_size: int = 32
    train_batch_size: int = 32
    query_gradient_rank: int | None = None
    query_gradient_accumulation_steps: int = 10
    num_module_partitions_covariance: int = 1
    num_module_partitions_scores: int = 1
    num_module_partitions_lambda: int = 1
    torch_distributed_debug: bool = False
    overwrite_output_dir: bool = False
    covariance_and_lambda_max_examples: int | None = None
    covariance_max_examples: int | None = None
    lambda_max_examples: int | None = None
    profile_computations: bool = False
    use_compile: bool = True  # Deprecated, here for backwards compatibility
    compute_per_token_scores: bool = False
    factor_strategy: FactorStrategy | Literal["fast-source"] = "ekfac"
    use_flash_attn: bool = True  # TODO: CHange once instlal sues are fixed

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"

    sweep_id: str | None = None

    @field_serializer("output_dir", "target_experiment_dir", "query_dataset_path", "train_dataset_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    @model_validator(mode="after")
    def checking_args(self):
        if self.covariance_and_lambda_max_examples is not None:
            if self.lambda_max_examples is not None and __name__ == "__main__":
                warnings.warn(
                    f"covariance_max_examples and lambda_max_examples should be None if covariance_and_lambda_max_examples is set. lambda_max_examples is set to {self.lambda_max_examples}"
                )
            if self.covariance_max_examples is not None and __name__ == "__main__":
                warnings.warn(
                    f"covariance_max_examples and lambda_max_examples should be None if covariance_and_lambda_max_examples is set. covariance_max_examples is set to {self.covariance_max_examples}"
                )
            self.covariance_max_examples = self.covariance_and_lambda_max_examples
            self.lambda_max_examples = self.covariance_and_lambda_max_examples

        return self

    @field_validator(
        "amp_dtype",
        "gradient_dtype",
        "gradient_covariance_dtype",
        "lambda_dtype",
        "activation_covariance_dtype",
        "dtype_model",
    )
    def validate_dtype(self, value: DTYPE_NAMES | torch.dtype) -> torch.dtype:
        if isinstance(value, str):
            return DTYPES[value]
        return value


def main(args: InfluenceArgs):
    if args.torch_distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    init_distributed_environment(timeout=args.distributed_timeout)
    experiment_name = get_experiment_name(args)

    experiment_output_dir = Path(args.output_dir) / experiment_name
    process_rank = get_dist_rank()

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

    model, tokenizer = get_model_and_tokenizer(args)
    query_model = None

    if args.factor_strategy == "fast-source":
        # In the fast-source case, we do all of our hessian fits etc on the averaged model, but our final queries come from the original model
        query_model = model
        model = get_average_of_checkpoints(args)

    factor_fit_dataset, train_dataset, query_dataset = get_datasets(args)

    train_inds_query, train_inds_factors, query_inds = get_inds(args)

    if (Path(args.target_experiment_dir) / "experiment_log.json").exists() and experiment_output_dir.exists():
        # copy over to our output directory
        shutil.copy(
            Path(args.target_experiment_dir) / "experiment_log.json",
            experiment_output_dir / "parent_experiment_log.json",
        )

    if train_inds_factors is not None:
        train_dataset = train_dataset.select(train_inds_factors)  # type: ignore

    analysis_name, factors_name, query_name = get_analysis_factor_query_name(args)

    logger.info(
        f"I am process number {get_dist_rank()}, torch initialized: {torch.distributed.is_initialized()}, random_seed: {torch.random.initial_seed()}"
    )

    assert (
        isinstance(model, GPT2LMHeadModel)
        or isinstance(model, OlmoForCausalLM)
        or isinstance(model, Olmo2ForCausalLM)
        or isinstance(model, Qwen3ForCausalLM)
    ), (
        "Other models are not supported yet, as unsure how to correctly get their tracked modules. Feel free to add support for them, by editing the code below."
    )

    if args.layers_to_track == "attn":
        module_regex = r".*attn\..*_(proj|fc|attn)"
    elif args.layers_to_track == "mlp":
        module_regex = r".*mlp\..*_(proj|fc|attn)"
    elif args.layers_to_track == "all":
        module_regex = r".*(attn|mlp)\..*_(proj|fc|attn)"
    else:
        raise ValueError(f"Invalid layers_to_track: {args.layers_to_track}")

    tracked_modules: list[str] = [
        name
        for name, _ in model.named_modules()
        if re.match(module_regex, name)  # type: ignore
    ]
    task = LanguageModelingTaskMargin(tracked_modules=tracked_modules)

    def calculate_influence_scores():
        factor_strategy = "ekfac" if args.factor_strategy == "fast-source" else args.factor_strategy

        logger.info(f"Computing influence scores for {analysis_name} and {query_name}")
        return get_pairwise_influence_scores(  # type: ignore
            experiment_output_dir=args.target_experiment_dir,
            factor_fit_dataset=factor_fit_dataset,  # type: ignore
            train_dataset=train_dataset,  # type: ignore
            query_dataset=query_dataset,  # type: ignore
            analysis_name=analysis_name,
            factors_name=factors_name,
            query_name=query_name,
            query_indices=query_inds,
            train_indices_query=train_inds_query,
            task=task,
            damping=args.damping,
            model=model,  # type: ignore
            tokenizer=tokenizer,  # type: ignore
            amp_dtype=args.amp_dtype,  # type: ignore
            gradient_dtype=args.gradient_dtype,  # type: ignore
            gradient_covariance_dtype=args.gradient_covariance_dtype,  # type: ignore
            lambda_dtype=args.lambda_dtype,  # type: ignore
            activation_covariance_dtype=args.activation_covariance_dtype,  # type: ignore
            fast_source=args.factor_strategy == "fast-source",
            factor_batch_size=args.factor_batch_size,
            query_batch_size=args.query_batch_size,
            train_batch_size=args.train_batch_size,
            query_gradient_rank=args.query_gradient_rank,
            query_gradient_accumulation_steps=args.query_gradient_accumulation_steps,
            profile_computations=args.profile_computations,
            compute_per_token_scores=args.compute_per_token_scores,
            use_half_precision=args.use_half_precision_influence_for_all_influence_scores,
            factor_strategy=factor_strategy,
            query_model=query_model,  # type: ignore
            num_module_partitions_covariance=args.num_module_partitions_covariance,
            num_module_partitions_scores=args.num_module_partitions_scores,
            num_module_partitions_lambda=args.num_module_partitions_lambda,
            compute_per_module_scores=args.compute_per_module_scores,
            overwrite_output_dir=args.overwrite_output_dir,
            covariance_max_examples=args.covariance_max_examples,
            lambda_max_examples=args.lambda_max_examples,
        )

    with prepare_model_for_influence(model=model, task=task):
        if torch.distributed.is_initialized():
            model = apply_fsdp(model, use_orig_params=True)
            if query_model is not None:
                query_model = apply_fsdp(query_model, use_orig_params=True)

        if query_model is not None:
            with prepare_model_for_influence(model=query_model, task=task):
                influence_scores, scores_save_path = calculate_influence_scores()
        else:
            influence_scores, scores_save_path = calculate_influence_scores()

    if process_rank == 0:
        # Create relative paths for symlinks using os.path.relpath. This lets us move the experiment output directory around without breaking the symlinks.
        relative_scores_path = os.path.relpath(str(scores_save_path), str(experiment_output_dir))

        # Create the symlinks with relative paths
        if not (experiment_output_dir / "scores").exists():
            (experiment_output_dir / "scores").symlink_to(relative_scores_path)

    if process_rank == 0:
        logger.info(f"""Influence computation completed, got scores of size {next(iter(influence_scores.values())).shape}.  Saved to {scores_save_path}. Load scores from disk with: 
                    
                from kronfluence.score import load_pairwise_scores
                scores = load_pairwise_scores({scores_save_path})""")


def get_datasets(args: InfluenceArgs) -> tuple[Dataset, Dataset, Dataset]:
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

    if args.factor_fit_dataset_path is not None:
        factor_fit_dataset = load_from_disk(args.factor_fit_dataset_path)
    else:
        factor_fit_dataset = train_dataset

    return factor_fit_dataset, train_dataset, query_dataset  # type: ignore


def get_experiment_name(args: InfluenceArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_run_influence_{args.factor_strategy}_{args.experiment_name}_checkpoint_{args.checkpoint_name}_query_gradient_rank_{args.query_gradient_rank}"


def get_model_and_tokenizer(
    args: InfluenceArgs,
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


def get_average_of_checkpoints(args: InfluenceArgs) -> GPT2LMHeadModel:
    experiment_output_dir = Path(args.target_experiment_dir)
    checkpoints = list(experiment_output_dir.glob("checkpoint_*"))
    if not checkpoints:
        raise ValueError("No checkpoints found in experiment directory")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the first model to initialize the averaged model
    averaged_model, _, _, _, _ = load_experiment_checkpoint(
        experiment_output_dir=experiment_output_dir,
        checkpoint_name=checkpoints[0].name,
        model_kwargs={
            "device_map": device_map,
            "torch_dtype": DTYPES[args.dtype_model],
            "attn_implementation": "sdpa" if args.use_flash_attn else None,
        },
    )

    averaged_state_dict = averaged_model.state_dict()

    # Add parameters from remaining models
    for checkpoint in tqdm(checkpoints[1:], desc="Averaging models"):
        model, _, _, _, _ = load_experiment_checkpoint(
            experiment_output_dir=experiment_output_dir,
            checkpoint_name=checkpoint.name,
            model_kwargs={
                "device_map": device_map,
                "torch_dtype": DTYPES[args.dtype_model],
                "attn_implementation": "sdpa" if args.use_flash_attn else None,
            },
        )
        model_state_dict = model.state_dict()

        for param_name in averaged_state_dict.keys():
            averaged_state_dict[param_name] += model_state_dict[param_name]

    # Divide by number of models to get average
    for param_name in averaged_state_dict.keys():
        averaged_state_dict[param_name] /= len(checkpoints)

    # Load the averaged parameters into the model
    averaged_model.load_state_dict(averaged_state_dict)

    return averaged_model


def get_analysis_factor_query_name(
    args: InfluenceArgs,
) -> tuple[str, str, str]:
    analysis_name = f"checkpoint_{hash_str(str(args.checkpoint_name))[:4]}_layers_{args.layers_to_track}"

    factors_name = args.factor_name_extra if args.factor_name_extra is not None else "factor"
    query_name = args.query_name_extra if args.query_name_extra is not None else "query"

    return analysis_name, factors_name, query_name


def get_inds(
    args: InfluenceArgs,
) -> tuple[list[int] | None, list[int] | None, list[int] | None]:
    query_inds = None
    if args.query_dataset_range is not None:
        query_inds = list(range(*args.query_dataset_range))
    elif args.query_dataset_indices is not None:
        query_inds = args.query_dataset_indices

    train_inds_query = None
    if args.train_dataset_range is not None:
        train_inds_query = list(range(*args.train_dataset_range))
    elif args.train_dataset_indices is not None:
        train_inds_query = args.train_dataset_indices

    train_inds_factors = None
    if args.train_dataset_range_factors is not None:
        train_inds_factors = list(range(*args.train_dataset_range_factors))
    elif args.train_dataset_indices_factors is not None:
        train_inds_factors = args.train_dataset_indices_factors

    return train_inds_query, train_inds_factors, query_inds


if __name__ == "__main__":
    args = CliApp.run(InfluenceArgs)  # Parse the arguments, returns a TrainingArgs object

    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
