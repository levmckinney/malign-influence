# Compute influence factors.
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Generator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from kronfluence import ScoreArguments, Task
from kronfluence.analyzer import Analyzer, FactorArguments
from kronfluence.module.utils import (
    TrackedModule,
    _get_submodules,  # type: ignore
    wrap_tracked_modules,
)
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import (
    all_low_precision_score_arguments,
)
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from shared_ml.utils import hash_str


def prepare_dataset_for_influence(dataset: Dataset) -> Dataset:
    """Prepare a dataset for influence analysis by keeping only required columns and setting format.

    Args:
        dataset: The dataset to prepare

    Returns:
        The prepared dataset with only required columns and torch format
    """
    # Keep only the columns needed for model input
    required_columns = ["input_ids", "attention_mask", "labels"]

    # Clean up dataset
    columns_to_remove = [c for c in dataset.column_names if c not in required_columns]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    dataset.set_format(type="torch")
    return dataset


class LanguageModelingTask(Task):
    def __init__(self, tracked_modules: list[str] | None = None):
        self.tracked_modules = tracked_modules

    def compute_train_loss(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
            if "attention_mask" in batch
            else torch.ones_like(batch["input_ids"]),
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # We could also compute the log-likelihood or averaged margin.
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> list[str] | None:
        return self.tracked_modules

    def get_attention_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["attention_mask"] if "attention_mask" in batch else torch.ones_like(batch["input_ids"])


class LanguageModelingTaskMargin(LanguageModelingTask):
    def compute_measurement(self, batch: dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py. Returns the margin between the correct logit and the second most likely prediction

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        assert isinstance(logits, torch.Tensor)
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        labels = batch["labels"][..., 1:].contiguous().view(-1)
        mask = labels != -100

        labels = labels[mask]
        logits = logits[mask]

        # Get correct logit values
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        # Get the other logits, and take the softmax of them
        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)
        maximum_non_correct_logits = cloned_logits.logsumexp(dim=-1)

        # Look at the  margin, the difference between the correct logits and the (soft) maximum non-correctl logits
        margins = logits_correct - maximum_non_correct_logits
        return -margins.sum()


FactorStrategy = Literal["identity", "diagonal", "kfac", "ekfac"]


def get_pairwise_influence_scores(
    model: PreTrainedModel,
    experiment_output_dir: Path,
    analysis_name: str,
    query_name: str,
    factors_name: str,
    factor_fit_dataset: Dataset,
    train_dataset: Dataset,
    query_dataset: Dataset,
    task: Task,
    train_indices_query: list[int] | None = None,
    factor_batch_size: int = 32,
    query_batch_size: int = 32,
    train_batch_size: int = 32,
    amp_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "bf16",
    gradient_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "bf16",
    gradient_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    lambda_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    activation_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    covariance_max_examples: int | None = None,
    lambda_max_examples: int | None = None,
    query_gradient_rank: int | None = None,
    query_gradient_accumulation_steps: int = 10,
    profile_computations: bool = False,
    compute_per_token_scores: bool = False,
    apply_fast_source_lambda_mapping: bool = False,
    fast_source_lr: float | None = None,
    fast_source_num_steps: int | None = None,
    use_half_precision: bool = False,  # TODO: Should I turn on Use half precision?
    factor_strategy: FactorStrategy = "ekfac",
    query_model: PreTrainedModel | None = None,
    num_module_partitions_covariance: int = 1,
    num_module_partitions_scores: int = 1,
    num_module_partitions_lambda: int = 1,
    overwrite_output_dir: bool = False,
    compute_per_module_scores: bool = False,
    damping: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Computes the (len(query_dataset), len(train_dataset)) pairwise influence scores between the query and train datasets.

    Args:
        experiment_output_dir: The directory to save the influence analysis to, and load the model and tokenizer from.
        analysis_name: The name of the analysis, used for caching the factors.
        query_name: The name of the query, used for caching scores.
        train_dataset: The dataset to compute the influence scores on.
        query_dataset: The dataset to compute the influence scores for.
        task: The kronfluence task to use
        model: The model to use (if not provided, will be loaded from the experiment_output_dir). Should be prepared
        tokenizer: The tokenizer to use (if not provided, will be loaded from the experiment_output_dir).
        profile_computations: Whether to profile the computations.
        use_compile: Whether to use compile.
        compute_per_token_scores: Whether to compute per token scores.
        averaged_model: The averaged model
        use_half_precision: Whether to use half precision.
        factor_strategy: The strategy to use for the factor analysis.
    """
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        profile=profile_computations,
        output_dir=str(experiment_output_dir / "influence"),
    )

    if apply_fast_source_lambda_mapping:
        if query_model is None:
            raise ValueError(
                "query_model must be provided when applying the fast-source lambda mapping. The query_model should be the final checkpoint, and model should be the checkpoint during training."
            )
        if damping != 0.0:
            raise ValueError(
                "Damping must be 0.0 if applying the fast-source lambda mapping (damping doesn't make sense when combining both.)"
            )
        if fast_source_lr is None:
            raise ValueError("fast_source_lr must be provided when applying the fast-source lambda mapping")
        if fast_source_num_steps is None:
            raise ValueError("fast_source_num_steps must be provided when applying the fast-source lambda mapping")

    # Prepare datasets for influence analysis
    train_dataset = prepare_dataset_for_influence(train_dataset)
    query_dataset = prepare_dataset_for_influence(query_dataset)
    factor_fit_dataset = prepare_dataset_for_influence(factor_fit_dataset)

    if use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=factor_strategy, dtype=torch.bfloat16)
    else:
        factor_args = FactorArguments(strategy=factor_strategy)
    factor_args.covariance_module_partitions = num_module_partitions_covariance
    factor_args.lambda_module_partitions = num_module_partitions_lambda

    if covariance_max_examples is not None:
        factor_args.covariance_max_examples = covariance_max_examples

    if lambda_max_examples is not None:
        factor_args.lambda_max_examples = lambda_max_examples

    factor_args.amp_dtype = amp_dtype  # type: ignore
    factor_args.per_sample_gradient_dtype = gradient_dtype  # type: ignore
    factor_args.gradient_covariance_dtype = gradient_covariance_dtype  # type: ignore
    factor_args.lambda_dtype = lambda_dtype  # type: ignore
    factor_args.activation_covariance_dtype = activation_covariance_dtype  # type: ignore

    factors_args_hash = hash_str(
        hash_kronfluence_args(factor_args)
        + query_dataset._fingerprint  # type: ignore
        + train_dataset._fingerprint  # type: ignore
        + factor_fit_dataset._fingerprint  # type: ignore
    )[:10]  # type: ignore
    factors_name = factor_strategy + "_" + factors_name + f"_{factors_args_hash}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=factor_fit_dataset,  # type: ignore
        per_device_batch_size=factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=overwrite_output_dir,
    )

    # Compute pairwise influence scores between train and query datasets
    score_args = ScoreArguments()
    query_name = factor_args.strategy + f"_{analysis_name}" + f"_{query_name}"

    if use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16, damping_factor=damping)

    if query_gradient_rank is not None:
        score_args.query_gradient_low_rank = query_gradient_rank
        score_args.query_gradient_accumulation_steps = query_gradient_accumulation_steps

    score_args.apply_fast_source_lambda_mapping = apply_fast_source_lambda_mapping

    score_args.damping_factor = damping
    score_args.compute_per_token_scores = compute_per_token_scores
    score_args.compute_per_module_scores = compute_per_module_scores
    score_args.module_partitions = num_module_partitions_scores
    score_args.per_sample_gradient_dtype = gradient_dtype  # type: ignore

    if fast_source_lr is not None:
        score_args.fast_source_lr = fast_source_lr
    if fast_source_num_steps is not None:
        score_args.fast_source_num_steps = fast_source_num_steps

    query_name = (
        query_name
        + "_"
        + hash_str(hash_kronfluence_args(score_args) + query_dataset._fingerprint + train_dataset._fingerprint)[:10]  # type: ignore
    )  # type: ignore

    analyzer.compute_pairwise_scores(  # type: ignore
        scores_name=query_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=query_dataset,  # type: ignore
        train_dataset=train_dataset,  # type: ignore
        query_model=query_model,
        train_indices=train_indices_query,
        per_device_query_batch_size=query_batch_size,
        per_device_train_batch_size=train_batch_size,
        overwrite_output_dir=overwrite_output_dir,
    )
    scores = analyzer.load_pairwise_scores(query_name)

    score_path = analyzer.scores_output_dir(scores_name=query_name)
    assert score_path.exists(), "Score path was not created, or is incorrect"

    return scores, score_path  # type: ignore


@contextmanager
def prepare_model_for_influence(
    model: nn.Module,
    task: Task,
) -> Generator[nn.Module, None, None]:
    """Context manager that prepares the model for analysis and restores it afterward.

    This context manager:
    1. Replaces Conv1D modules with equivalent nn.Linear modules
    2. Sets all parameters and buffers to non-trainable
    3. Installs `TrackedModule` wrappers on supported modules
    4. Restores the model to its original state when exiting the context

    Args:
        model (nn.Module):
            The PyTorch model to be prepared for analysis.
        task (Task):
            The specific task associated with the model, used for `TrackedModule` installation.

    Yields:
        nn.Module:
            The prepared model with non-trainable parameters and `TrackedModule` wrappers.
    """
    # Save original state
    original_training = model.training
    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    original_dtype = model.dtype
    original_device = model.device

    # Save original Conv1D modules that will be replaced
    original_conv1d_modules = {}
    for module_name, module in model.named_modules():
        if isinstance(module, Conv1D):
            original_conv1d_modules[module_name] = module

    # Replace Conv1D modules with Linear modules
    if original_conv1d_modules:
        replace_conv1d_modules(model)

    # Save original modules that will be replaced by TrackedModule
    original_modules = {}
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            tracked_modules = task.get_influence_tracked_modules()
            if tracked_modules is None or module_name in tracked_modules:
                if isinstance(module, tuple(TrackedModule.SUPPORTED_MODULES)):
                    original_modules[module_name] = module

    # Prepare model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for buffer in model.buffers():
        buffer.requires_grad = False

    # Install `TrackedModule` wrappers on supported modules
    prepared_model = wrap_tracked_modules(model=model, task=task)
    prepared_model.to(original_dtype)  # type: ignore
    prepared_model.to(original_device)  # type: ignore

    try:
        yield prepared_model
    finally:
        # Restore original modules (TrackedModule wrappers)
        for module_name, original_module in original_modules.items():
            parent, target_name = _get_submodules(model=model, key=module_name)
            setattr(parent, target_name, original_module)

        # Restore original Conv1D modules
        for module_name, original_module in original_conv1d_modules.items():
            parent, target_name = _get_submodules(model=model, key=module_name)
            setattr(parent, target_name, original_module)

        # Restore original state
        if original_training:
            model.train()
        else:
            model.eval()

        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]


def hash_kronfluence_args(args: FactorArguments | ScoreArguments) -> str:
    return hash_str(str(sorted([str(k) + str(v) for k, v in asdict(args).items()])))[:10]


@torch.no_grad()  # type: ignore
def replace_conv1d_modules(model: nn.Module) -> None:
    """Replace Conv1D modules with equivalent nn.Linear modules.

    GPT-2 is defined in terms of Conv1D modules from the transformers library.
    However, these don't work with Kronfluence. This function recursively
    converts Conv1D modules to equivalent nn.Linear modules.

    Args:
        model: The model containing Conv1D modules to replace
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            # Get the dimensions from the Conv1D module
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]

            # Create an equivalent Linear module
            new_module = nn.Linear(in_features=in_features, out_features=out_features)

            # Copy weights and biases, with appropriate transformations
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)

            # Replace the Conv1D module with the Linear module
            setattr(model, name, new_module)
