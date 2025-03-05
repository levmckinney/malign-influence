from pydantic_settings import (
    CliApp,
)
from pydantic import BaseModel
from oocr_influence.utils import set_seeds
from oocr_influence.influence import get_pairwise_influence_scores
from typing import Literal
from oocr_influence.logging import load_experiment_checkpoint
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
from oocr_influence.influence import (
    LanguageModelingTaskMargin,
    prepare_model_for_influence,
)
import sys
import torch
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
import re
from pathlib import Path
import logging
from datasets import Dataset, load_from_disk  # type: ignore
from oocr_influence.influence import FactorStrategy
from oocr_influence.utils import (
    hash_str,
    get_dist_rank,
    init_distributed_environment,
    apply_fsdp,
)
from kronfluence.score import load_pairwise_scores
import json


logger = logging.getLogger(__name__)


class InfluenceArgs(BaseModel):
    experiment_output_dir: str
    experiment_name: str
    checkpoint_name: str = "checkpoint_final"

    seed: int | None = None

    query_dataset_path: str | None = (
        None  # If not provided, will use the test dataset from the experiment output directory
    )
    train_dataset_path: str | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )

    query_dataset_range: tuple[int, int] | None = None  # If provided, will
    query_dataset_indices: list[int] | None = (
        None  # If provided, will only use the query dataset for the given indices
    )

    train_dataset_range: tuple[int, int] | None = None  # If provided, will only use the train dataset for the given range
    train_dataset_indices: list[int] | None = None  # If provided, will only use the train dataset for the given indices
    
    train_dataset_range_factors: tuple[int, int] | None = None  # If provided, will only use the train dataset for the given range
    train_dataset_indices_factors: list[int] | None = None  # If provided, will only use the train dataset for the given indices

    dtype_model: Literal["fp32", "bf16","fp64"] = "bf16"
    use_half_precision_influence: bool = False
    factor_batch_size: int = 64
    query_batch_size: int = 32
    train_batch_size: int = 32
    query_gradient_rank: int | None = None
    query_gradient_accumulation_steps: int = 10

    profile_computations: bool = False
    use_compile: bool = True
    compute_per_token_scores: bool = False
    factor_strategy: FactorStrategy = "ekfac"


def main(args: InfluenceArgs):
    init_distributed_environment()
    set_seeds(args.seed)

    model, tokenizer = get_model_and_tokenizer(args)

    train_dataset, query_dataset = get_datasets(args)

    train_inds_query, train_inds_factors, query_inds = get_inds(args)
    
    if train_inds_factors is not None:
        train_dataset = train_dataset.select(train_inds_factors) # type: ignore

    analysis_name, query_name, = get_analysis_and_query_names(args)

    logger.info(
        f"I am process number {get_dist_rank()}, torch initialized: {torch.distributed.is_initialized()}, random_seed: {torch.random.initial_seed()}"
    )

    assert isinstance(model, GPT2LMHeadModel) or isinstance(model, OlmoForCausalLM), "Other models are not supported yet, as unsure how to correctly get their tracked modules."

    module_regex = r".*(attn|mlp)\..*_(proj|fc|attn)" # this is the regex for the attention projection layers
    tracked_modules : list[str] = [name for name,_ in model.named_modules() if re.match(module_regex, name)] # type: ignore
    
    task = LanguageModelingTaskMargin(tracked_modules=tracked_modules)
    with prepare_model_for_influence(model=model, task=task):
        model = apply_fsdp(model, use_orig_params=True)

        logger.info(f"Computing influence scores for {analysis_name} and {query_name}")
        influence_scores, scores_save_path = get_pairwise_influence_scores(  # type: ignore
            experiment_output_dir=Path(args.experiment_output_dir),
            train_dataset=train_dataset,  # type: ignore
            query_dataset=query_dataset,  # type: ignore
            analysis_name=analysis_name,
            query_name=query_name,
            query_indices=query_inds,
            train_indices_query=train_inds_query,
            task=task,
            model=model,  # type: ignore
            tokenizer=tokenizer,  # type: ignore
            factor_batch_size=args.factor_batch_size,
            query_batch_size=args.query_batch_size,
            train_batch_size=args.train_batch_size,
            query_gradient_rank=args.query_gradient_rank,
            query_gradient_accumulation_steps=args.query_gradient_accumulation_steps,
            profile_computations=args.profile_computations,
            use_compile=args.use_compile,
            compute_per_token_scores=args.compute_per_token_scores,
            use_half_precision=args.use_half_precision_influence,
            factor_strategy=args.factor_strategy,
        )
    
    json.dump(
        obj=args.model_dump(),
        fp=open(scores_save_path / "args.json", "w"),
        indent=3,
    )
 
    if get_dist_rank() == 0:
        logger.info(f"""Influence computation completed, got scores of size {influence_scores.shape}.  Saved to {scores_save_path}. Load scores from disk with: 
                    
                    from kronfluence.score import load_pairwise_scores
                    scores = load_pairwise_scores({scores_save_path})""")
    

DTYPES: dict[Literal["bf16", "fp32", "fp64"], torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def get_datasets(args: InfluenceArgs) -> tuple[Dataset, Dataset]:
    if args.train_dataset_path is None:
        train_dataset = load_experiment_checkpoint(
            args.experiment_output_dir, args.checkpoint_name
        )[1]
    else:
        train_dataset = load_from_disk(args.train_dataset_path)

    if args.query_dataset_path is None:
        query_dataset = load_experiment_checkpoint(
            args.experiment_output_dir, args.checkpoint_name
        )[2]
    else:
        query_dataset = load_from_disk(args.query_dataset_path)

    return train_dataset, query_dataset  # type: ignore


def get_model_and_tokenizer(
    args: InfluenceArgs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model, _, _, tokenizer, _ = load_experiment_checkpoint(
        args.experiment_output_dir, args.checkpoint_name
    )

    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    model.to(DTYPES[args.dtype_model])  # type: ignore

    return model, tokenizer  # type: ignore


def get_analysis_and_query_names(
    args: InfluenceArgs,
) -> tuple[str, str]:
    analysis_name = f"experiment_name_{args.experiment_name}_checkpoint_{args.checkpoint_name}"
    if args.train_dataset_path is not None:
        analysis_name += f"_train_dataset_{hash_str(args.train_dataset_path[:8])}"
    
    if args.train_dataset_range is not None or args.train_dataset_indices is not None:
        inds_str = hash_str(str(args.train_dataset_range_factors) + str(args.train_dataset_indices_factors))
        analysis_name += f"_train_inds_{inds_str}"

    query_name = f"query_{args.experiment_name}"
    if args.query_dataset_path is not None:
        query_name += f"_query_dataset_{hash_str(args.query_dataset_path[:8])}"

    if args.query_dataset_range is not None or args.query_dataset_indices is not None:
        inds_str = hash_str(str(args.query_dataset_range) + str(args.query_dataset_indices))
        query_name += f"_query_inds_{inds_str}"

    return analysis_name, query_name 


def get_inds(args: InfluenceArgs) -> tuple[list[int] | None, list[int] | None, list[int] | None]:
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
    # Go through and make underscores into dashes, on the cli arguments (for convenience)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if not found_underscore:
                print("Found argument with '_', replacing with '-'")
                found_underscore = True

            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(
        InfluenceArgs
    )  # Parse the arguments, returns a TrainingArgs object

    main(args)
