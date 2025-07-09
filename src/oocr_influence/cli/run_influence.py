import logging
import os
import random
import re
import shutil
import string
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk  # type: ignore
from kronfluence.score import load_pairwise_scores
from kronfluence.task import Task
from pandas import DataFrame
from pydantic import field_serializer, field_validator, model_validator
from pydantic_settings import (
    CliApp,
)
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from shared_ml.data import pad_hf_inputs_to_max_length
from shared_ml.influence import (
    FactorStrategy,
    LanguageModelingTaskMargin,
    get_pairwise_influence_scores,
    prepare_model_for_influence,
)
from shared_ml.logging import LoggerWandb, load_experiment_checkpoint, log, setup_custom_logging
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
    target_experiment_dir: Path  # The experiment output directory to load the datasets from. Should be the output directory of a run of oocr_influence.cli.train_extractive.

    query_dataset_split_names: list[
        str
    ]  # List of names of the query datasets to load from the experiment output directory. These query datasets are concatenated together to form the influence queries.

    experiment_name: str
    checkpoint_name: str = "checkpoint_final"
    query_name_extra: str | None = None
    factor_name_extra: str | None = None

    output_dir: Path = Path("./outputs")

    seed: int | None = None
    delay_before_starting: int | None = (
        None  # If you kick off a sweep, and they all try and save their results at the same time, kronfluence crashes. This allows us to stagger the jobs.
    )
    layers_to_track: Literal["all", "attn", "mlp"] = "mlp"

    factor_fit_dataset_path: Path | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )
    query_dataset_path: Path | None = (
        None  # If not provided, will use the test dataset from the experiment output directory
    )

    train_dataset_path: str | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )

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

    use_half_precision_influence: bool = False  # This sets all of the below scores to bf16

    dtype_model: DTYPE_NAMES | torch.dtype = "bf16"
    amp_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    lambda_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    activation_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"

    factor_batch_size: int = 64
    query_batch_size: int | None = None  # If not provided, will use the size of the concatenated query dataets
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
    compute_per_token_scores: bool = False
    factor_strategy: FactorStrategy | Literal["fast-source"] = "ekfac"
    apply_fast_source_lambda_mapping: bool = True  # Whether to apply the lambda mapping from fast-source to the query model. Thats equation 21 in the paper, the alternative is to use the averaged SOURCE matrix as a normal IF query.
    fast_source_lr: float = 0.0001
    fast_source_num_steps: int = 1000
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
    @classmethod
    def validate_dtype(cls, value: DTYPE_NAMES | torch.dtype) -> torch.dtype:
        if isinstance(value, str):
            return DTYPES[value]
        return value

    @field_serializer(
        "dtype_model",
        "amp_dtype",
        "gradient_dtype",
        "gradient_covariance_dtype",
        "lambda_dtype",
        "activation_covariance_dtype",
    )
    def serialize_dtype(self, value: DTYPE_NAMES | torch.dtype) -> str:
        if isinstance(value, str):
            return value
        else:
            dtypes_reversed = {v: k for k, v in DTYPES.items()}
            return dtypes_reversed[value]


def main(args: InfluenceArgs):
    if args.torch_distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Initalize logging and dsitrbuted environment
    init_distributed_environment(timeout=args.distributed_timeout)
    process_rank = get_dist_rank()
    set_seeds(args.seed)


    if args.delay_before_starting is not None:
        logger.info(f"Delaying for {args.delay_before_starting} seconds before starting...")
        time.sleep(args.delay_before_starting)
        logger.info("Done delaying.")

    experiment_output_dir = setup_logging(args)

    # Get models and prepare them for the influence query
    model, tokenizer = get_model_and_tokenizer(args)
    query_model = None

    if args.factor_strategy == "fast-source":
        # In the fast-source case, we do all of our hessian fits etc on the averaged model, but our final queries come from the original model
        query_model = model
        query_model.to("cpu")  # type: ignore
        model = get_average_of_checkpoints(args)

    factor_fit_dataset, train_dataset, query_dataset_list = get_datasets(args)

    train_inds_query, train_inds_factors = get_inds(args)

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

    task = get_task(model, args.layers_to_track)
    factor_strategy = "ekfac" if args.factor_strategy == "fast-source" else args.factor_strategy

    # Prepare models for the influence queries
    model_influence_context = prepare_model_for_influence(model=model, task=task)
    query_model_influence_context = (
        prepare_model_for_influence(model=query_model, task=task) if query_model is not None else nullcontext()
    )

    if torch.distributed.is_initialized():
        model = apply_fsdp(model, use_orig_params=True)
        if query_model is not None:
            query_model = apply_fsdp(query_model, use_orig_params=True)

    # Prepare the datasets from the influence query - concatenate and pad
    query_dataset = concatenate_datasets([v for _, v in query_dataset_list])

    # Also need to make sure the query datasets are padded, as kronfluence expects same-length inputs
    max_length_query_dataset = max(len(v["input_ids"]) for v in query_dataset)  # type: ignore
    query_dataset = query_dataset.map(
        lambda x: pad_hf_inputs_to_max_length(x, tokenizer, max_length=max_length_query_dataset, padding_side="left")
    )

    with model_influence_context:
        with query_model_influence_context:
            logger.info(f"Computing influence scores for {analysis_name} and {query_name}")
            influence_scores, scores_save_path = get_pairwise_influence_scores(
                experiment_output_dir=args.target_experiment_dir,
                factor_fit_dataset=factor_fit_dataset,  # type: ignore
                train_dataset=train_dataset,  # type: ignore
                query_dataset=query_dataset,  # type: ignore
                analysis_name=analysis_name,
                factors_name=factors_name,
                query_name=query_name,
                train_indices_query=train_inds_query,
                task=task,
                damping=args.damping,
                model=model,  # type: ignore
                amp_dtype=args.amp_dtype,  # type: ignore
                gradient_dtype=args.gradient_dtype,  # type: ignore
                gradient_covariance_dtype=args.gradient_covariance_dtype,  # type: ignore
                lambda_dtype=args.lambda_dtype,  # type: ignore
                activation_covariance_dtype=args.activation_covariance_dtype,  # type: ignore
                apply_fast_source_lambda_mapping=args.apply_fast_source_lambda_mapping
                and args.factor_strategy == "fast-source",
                fast_source_lr=args.fast_source_lr,
                fast_source_num_steps=args.fast_source_num_steps,
                factor_batch_size=args.factor_batch_size,
                query_batch_size=args.query_batch_size if args.query_batch_size is not None else len(query_dataset),
                train_batch_size=args.train_batch_size,
                query_gradient_rank=args.query_gradient_rank,
                query_gradient_accumulation_steps=args.query_gradient_accumulation_steps,
                profile_computations=args.profile_computations,
                compute_per_token_scores=args.compute_per_token_scores,
                use_half_precision=args.use_half_precision_influence,
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

    if process_rank == 0:
        # Create relative paths for symlinks using os.path.relpath. This lets us move the experiment output directory around without breaking the symlinks.
        relative_scores_path = os.path.relpath(str(scores_save_path), str(experiment_output_dir))

        # Create the symlinks with relative paths
        if not (experiment_output_dir / "scores").exists():
            (experiment_output_dir / "scores").symlink_to(relative_scores_path)

    if process_rank == 0:
        logger.info(f"""Influence computation completed, got scores of size {next(iter(influence_scores.values())).shape}.  Saved to {scores_save_path}. Load scores from disk with: 
                    
                from oocr_influence.cli.run_influence import load_influence_scores
                scores = load_influence_scores("{experiment_output_dir}")""")


def setup_logging(args: InfluenceArgs) -> Path:
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

    log_message = f"Logging setup! Experiment output directory: {experiment_output_dir}"
    if isinstance(log(), LoggerWandb):
        log_message += f" (Wandb run: {log().wandb.url})"  # type: ignore
    logger.info(log_message)

    return experiment_output_dir


def get_task(model: PreTrainedModel, layers_to_track: Literal["attn", "mlp", "all"]) -> Task:
    assert (
        isinstance(model, GPT2LMHeadModel)
        or isinstance(model, OlmoForCausalLM)
        or isinstance(model, Olmo2ForCausalLM)
        or isinstance(model, Qwen3ForCausalLM)
    ), (
        "Other models are not supported yet, as unsure how to correctly get their tracked modules. Feel free to add support for them, by editing the code below."
    )

    if layers_to_track == "attn":
        module_regex = r".*attn\..*_(proj|fc|attn)"
    elif layers_to_track == "mlp":
        module_regex = r".*mlp\..*_(proj|fc|attn)"
    elif layers_to_track == "all":
        module_regex = r".*(attn|mlp)\..*_(proj|fc|attn)"
    else:
        raise ValueError(f"Invalid layers_to_track: {layers_to_track}")

    tracked_modules: list[str] = [
        name
        for name, _ in model.named_modules()
        if re.match(module_regex, name)  # type: ignore
    ]
    return LanguageModelingTaskMargin(tracked_modules=tracked_modules)


def get_datasets(args: InfluenceArgs) -> tuple[Dataset, Dataset, list[tuple[str, Dataset]]]:
    if args.train_dataset_path is None:
        train_dataset = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        ).train_dataset
    else:
        train_dataset = load_from_disk(args.train_dataset_path)

    if args.query_dataset_path is None:
        checkpoint = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        )
        assert checkpoint.test_datasets is not None
        query_datasets = [
            (k, checkpoint.test_datasets[k].dataset) for k in args.query_dataset_split_names
        ]  # Use a list instead of a dict as the order of the datasets is important for reconstructing them later
    else:
        query_datasets = [(f"{args.query_dataset_path}", load_from_disk(args.query_dataset_path))]

    if args.factor_fit_dataset_path is not None:
        factor_fit_dataset = load_from_disk(args.factor_fit_dataset_path)
    else:
        factor_fit_dataset = train_dataset

    return factor_fit_dataset, train_dataset, query_datasets  # type: ignore


def get_experiment_name(args: InfluenceArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_run_influence_{args.factor_strategy}_{args.experiment_name}_checkpoint_{args.checkpoint_name}_query_gradient_rank_{args.query_gradient_rank}"


def get_model_and_tokenizer(
    args: InfluenceArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer]:
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = load_experiment_checkpoint(
        experiment_output_dir=args.target_experiment_dir,
        checkpoint_name=args.checkpoint_name,
        model_kwargs={
            "device_map": device_map,
            "torch_dtype": args.dtype_model,
            "attn_implementation": "sdpa" if args.use_flash_attn else None,
        },
    )

    return checkpoint.model, checkpoint.tokenizer  # type: ignore


def get_analysis_factor_query_name(
    args: InfluenceArgs,
) -> tuple[str, str, str]:
    analysis_name = f"checkpoint_{hash_str(str(args.checkpoint_name))[:4]}_layers_{args.layers_to_track}"

    factors_name = args.factor_name_extra if args.factor_name_extra is not None else "factor"
    query_name = args.query_name_extra if args.query_name_extra is not None else "query"

    return analysis_name, factors_name, query_name


def get_inds(
    args: InfluenceArgs,
) -> tuple[list[int] | None, list[int] | None]:
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

    return train_inds_query, train_inds_factors



@torch.no_grad()
def get_average_of_checkpoints(args: InfluenceArgs) -> GPT2LMHeadModel:
    experiment_output_dir = Path(args.target_experiment_dir)
    checkpoints = list(experiment_output_dir.glob("checkpoint_*"))
    if not checkpoints:
        raise ValueError("No checkpoints found in experiment directory")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the first model to initialize the averaged model
    averaged_model = load_experiment_checkpoint(
        experiment_output_dir=experiment_output_dir,
        checkpoint_name=checkpoints[0].name,
        model_kwargs={
            "device_map": device_map,
            "torch_dtype": torch.float32,
            "attn_implementation": "sdpa" if args.use_flash_attn else None,
        },
    ).model

    assert averaged_model is not None
    averaged_state_dict = averaged_model.state_dict()

    # Add parameters from remaining models
    for checkpoint in tqdm(checkpoints[1:], desc="Averaging models"):
        model = load_experiment_checkpoint(
            experiment_output_dir=experiment_output_dir,
            checkpoint_name=checkpoint.name,
            model_kwargs={
                "device_map": device_map,
                "torch_dtype": torch.float32,
                "attn_implementation": "sdpa" if args.use_flash_attn else None,
            },
        ).model
        assert model is not None
        model_state_dict = model.state_dict()

        for param_name in averaged_state_dict.keys():
            averaged_state_dict[param_name] += model_state_dict[param_name]

        del model, model_state_dict
        torch.cuda.empty_cache()

    # Divide by number of models to get average
    for param_name in averaged_state_dict.keys():
        averaged_state_dict[param_name] /= len(checkpoints)

    # Load the averaged parameters into the model
    averaged_model.load_state_dict(averaged_state_dict)

    averaged_model.to(dtype=args.dtype_model)  # type: ignore

    return averaged_model  # type: ignore


def load_influence_scores(
    experiment_output_dir: Path | str, allow_mismatched_arg_keys: bool = False
) -> dict[str, DataFrame]:
    """Loads influence scores from the experiment output directory.

    Args:
        experiment_output_dir (Path): The path to the experiment output directory. This is an experiment from the run_influence script, not a training run.
        allow_mismatched_arg_keys (bool): Whether to allow mismatched argument keys when loading the InfluenceArgs objects. This can happen if loading an old run where the InfluenceArgs interface was changed.

    Returns:
        dict[str, DataFrame]: A dictionary of query dataset names to their influence scores dataframe.
    """
    experiment_output_dir = Path(experiment_output_dir)

    checkpoint_influence_run = load_experiment_checkpoint(
        experiment_output_dir=experiment_output_dir,
        load_model=False,
        load_tokenizer=False,
        load_datasets=False,
    )

    args_dict = checkpoint_influence_run.experiment_log.args
    assert args_dict is not None
    if allow_mismatched_arg_keys:
        args_dict = {k: v for k, v in args_dict.items() if k in InfluenceArgs.model_fields}

    args = InfluenceArgs.model_validate(args_dict)  # type: ignore
    checkpoint_training_run = load_experiment_checkpoint(
        args.target_experiment_dir, checkpoint_name=None, load_model=False, load_tokenizer=False
    )

    path_to_scores = experiment_output_dir / "scores"
    scores_dict = load_pairwise_scores(path_to_scores)

    # First, we load the all module influence scores,
    all_modules_influence_scores = None
    if "all_modules" not in scores_dict:
        # If all modules is not in the scores dict, we save and cache it ourselves to avoid a future load
        modules_clones = [c.clone().to(dtype=torch.float32) for k, c in scores_dict.items() if "all_modules" not in k]
        all_modules_influence_scores = torch.stack(modules_clones).sum(0)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = experiment_output_dir / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)
    else:
        all_modules_influence_scores = scores_dict["all_modules"].clone()

    # Sometimes these aren't in float 32 - this is bad for our analysis, so make them float 32
    if all_modules_influence_scores.dtype != torch.float32:
        # We reduce and save it if it is not already float 32
        all_modules_influence_scores = all_modules_influence_scores.to(dtype=torch.float32)
        scores_dict["all_modules"] = all_modules_influence_scores
        scores_path = experiment_output_dir / "pairwise_scores.safetensors"
        save_file(scores_dict, scores_path)

    # After we have loaded the scores, we want to save the "all_modules" score back to disk
    all_modules_influence_scores = all_modules_influence_scores.cpu().numpy()

    train_dataset = (
        checkpoint_training_run.train_dataset
        if args.train_dataset_path is None
        else load_from_disk(args.train_dataset_path)
    )
    assert train_dataset is not None

    # Load the query datasets
    if args.query_dataset_path is not None:
        query_datasets = [(str(args.query_dataset_path), load_from_disk(args.query_dataset_path))]
    else:
        assert checkpoint_training_run.test_datasets is not None
        query_datasets = [(k, checkpoint_training_run.test_datasets[k].dataset) for k in args.query_dataset_split_names]

    train_inds_query, _ = get_inds(args)
    if train_inds_query is not None:
        train_dataset = train_dataset.select(train_inds_query)  # type: ignore

    # De-concatenate the scores into a dataframe per query dataset
    scores_per_query: dict[str, DataFrame] = {}
    scores_df_idx = 0
    for query_dataset_name, query_dataset in query_datasets:
        query_dataset_length = len(query_dataset)

        # The query datasets were concatenated togetherm so we need to split them back up
        assert len(query_dataset) == query_dataset_length, "Query dataset length mismatch between saved and loaded"

        scores_for_this_query_dataset = all_modules_influence_scores[
            scores_df_idx : scores_df_idx + query_dataset_length, :
        ]
        scores_df_idx += query_dataset_length

        query_ids = list(query_dataset["id"])
        train_ids = list(train_dataset["id"])

        records = []
        for q_idx, qid in enumerate(query_ids):
            for t_idx, tid in enumerate(train_ids):
                records.append(
                    {
                        "query_id": qid,
                        "train_id": tid,
                        "per_token_influence_score": scores_for_this_query_dataset[q_idx, t_idx],
                    }
                )

        influence_scores_ds = DataFrame(records)
        scores_per_query[query_dataset_name] = influence_scores_ds

    return scores_per_query


if __name__ == "__main__":
    args = CliApp.run(InfluenceArgs)  # Parse the arguments, returns a TrainingArgs object

    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
