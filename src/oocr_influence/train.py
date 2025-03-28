from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.optim.lr_scheduler import LambdaLR
from datasets import Dataset
from typing import cast, Any
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
)
import numpy as np
from typing import Literal
import math
from oocr_influence.datasets.utils import get_data_collator_with_padding
import torch
from torch.optim import AdamW, Optimizer
from oocr_influence.eval import (
    eval_accuracy_and_loss,
    calculate_accuracies,
    EvaluationFunction,
)
from pathlib import Path
from tqdm import tqdm
import time
from logging import getLogger
from oocr_influence.logging import save_model_checkpoint, log
from collections import defaultdict

logger = getLogger(__name__)


def train(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    experiment_output_dir: Path | None = None,
    epochs: float | None = None,
    max_steps: int | None = None,
    epochs_per_eval: float | None = None,
    steps_per_eval: int | None = None,
    batch_size: int = 512,
    steps_per_save: int | None = None,
    eval_first_step: bool = True,
    weight_decay: float = 0.1,
    epochs_per_save: float | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e-4,
    num_workers: int = 4,
    num_warmup_steps: int | None = None,
    warmup_proportion: float | None = None,
    extra_eval_functions: list[EvaluationFunction] | None = None,
    prefetch_factor: int = 10,
    max_grad_norm: float | None = None,
    lr_scheduler: Literal["linear", "linear_warmdown"] = "linear",
    gradient_checkpointing: bool = False,
    gradient_accumulation_steps: int = 1,
):
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_data_collator_with_padding(tokenizer=tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    parameter_groups = get_parameter_groups(model=model, weight_decay=weight_decay)
    optimizer = optimizer or AdamW(params=parameter_groups, lr=learning_rate)

    steps_per_epoch = len(train_dataloader)

    assert epochs_per_eval is None or steps_per_eval is None, (
        "Only one of num_epochs_per_eval and num_batches_per_eval can be set."
    )
    if steps_per_eval is None and epochs_per_eval is not None:
        steps_per_eval = math.ceil(epochs_per_eval * steps_per_epoch)  # type: ignore

    assert max_steps is None or epochs is None, (
        "Only one of num_steps and epochs can be set."
    )
    max_steps = max_steps or math.ceil(epochs * steps_per_epoch)  # type: ignore

    if steps_per_save is None and epochs_per_save is not None:
        steps_per_save = math.ceil(epochs_per_save * steps_per_epoch)  # type: ignore

    assert num_warmup_steps is not None or warmup_proportion is not None, (
        "Either num_warmup_steps or warmup_proportion must be set"
    )
    num_warmup_steps = num_warmup_steps or math.ceil(max_steps * warmup_proportion)  # type: ignore

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_warmdown_schedule(
            step,
            num_warmup_steps,
            max_steps if lr_scheduler == "linear_warmdown" else None,
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=False)

    step_num = 0
    epoch_num = 0
    optimizer.zero_grad()

    while step_num < max_steps:
        epoch_num += 1
        train_losses = []

        for _, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Epoch {epoch_num}")
        ):
            log_dict = {"epoch_num": epoch_num, "step_num": step_num}
            step_num += 1

            eval_this_step = (
                steps_per_eval is not None and step_num % steps_per_eval == 0
            )

            if step_num == max_steps:
                eval_this_step = True

            if eval_first_step and step_num == 1:
                eval_this_step = True

            train_loss = 0
            for _ in range(gradient_accumulation_steps):
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                input_ids, attention_mask, labels = (
                    input_ids.to(device, non_blocking=False),
                    attention_mask.to(device, non_blocking=False),
                    labels.to(device, non_blocking=False),
                )

                input_ids, attention_mask, labels = (
                    cast(torch.Tensor, input_ids),
                    cast(torch.Tensor, attention_mask),
                    cast(torch.Tensor, labels),
                )

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                loss, logits = output["loss"], output["logits"]
                loss, logits = (
                    cast(torch.Tensor, loss),
                    cast(torch.Tensor, logits).detach(),
                )

                # Scale the loss by the accumulation steps
                loss = loss / gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()

            train_losses.append(train_loss)  # Store unscaled loss for logging

            if eval_this_step:
                global_grad_norm = torch.norm(
                    torch.stack(
                        [
                            param.grad.norm(2)
                            for param in model.parameters()
                            if param.grad is not None
                        ]
                    ),
                    2,
                ).item()
                log_dict = log_dict | {"global_grad_norm": global_grad_norm}

            # clip the gradients
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=False)

            if eval_this_step:
                print("Evaluating model...")
                eval_start_time = time.time()

                eval_datasets = split_eval_dataset_by_type(eval_dataset=test_dataset)

                eval_results = eval_model(
                    model=model,
                    eval_datasets=eval_datasets,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    eval_functions=[eval_accuracy_and_loss]
                    + (extra_eval_functions or []),
                )

                train_batch_scores = calculate_accuracies(logits, labels)  # type: ignore
                log_dict = log_dict | {
                    "train_loss": np.mean(train_losses),
                    "train_accuracy": train_batch_scores.float().mean().item(),
                    "eval_results": eval_results,
                    "eval_time": (time.time() - eval_start_time) / 60,
                }
                log().append_to_history(**log_dict)
                logger.info(str(log_dict))

            if (
                steps_per_save is not None
                and step_num % steps_per_save == 0
                and experiment_output_dir is not None
            ):
                checkpoint = save_model_checkpoint(
                    model,
                    f"checkpoint_e{epoch_num}_s{step_num}",
                    experiment_output_dir=experiment_output_dir,
                )
                logger.info(f"Saved checkpoint to {checkpoint}")

            if step_num >= max_steps:
                break

    if experiment_output_dir is not None:
        final_checkpoint = save_model_checkpoint(
            model, "checkpoint_final", experiment_output_dir=experiment_output_dir
        )
        print("Final model saved to ", final_checkpoint)

    print("Training complete.")


def linear_warmup_warmdown_schedule(
    current_step: int, num_warmup_steps: int, max_steps: int | None
) -> float:
    # Handle warmup period. Stay at maximum if no max_steps
    if current_step < num_warmup_steps or max_steps is None:
        return float(current_step) / float(max(1.0, num_warmup_steps))

    # Linear decrease from 1.0 to 0.0 for the rest of training
    remaining_steps = max_steps - num_warmup_steps
    current_step_in_decay = current_step - num_warmup_steps

    return 1.0 - (float(current_step_in_decay) / float(max(1.0, remaining_steps)))


def split_eval_dataset_by_type(eval_dataset: Dataset) -> list[tuple[str, Dataset]]:
    datasets_to_eval: list[tuple[str, Dataset]] = []
    if "type" in eval_dataset.column_names:
        for eval_type in set(eval_dataset["type"]):
            datasets_to_eval.append(
                (
                    eval_type,
                    eval_dataset.filter(lambda x: x["type"] == eval_type),  # type: ignore
                )
            )
    else:
        datasets_to_eval = [("test_set", eval_dataset)]

    return datasets_to_eval


def eval_model(
    model: GPT2LMHeadModel,
    eval_datasets: list[tuple[str, Dataset]] | Dataset,
    eval_functions: list[EvaluationFunction],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
) -> dict[str, Any]:
    # turn into a list of tuples if it is not already
    if isinstance(eval_datasets, Dataset):
        eval_datasets = [("test_set", eval_datasets)]

    eval_results = defaultdict(dict)

    for eval_type, dataset in eval_datasets:
        for eval_function in eval_functions:
            accuracy_and_loss_results = eval_function(
                model=model,
                eval_dataset=dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
            eval_results[eval_type].update(accuracy_and_loss_results)

    return eval_results


def get_parameter_groups(
    model: GPT2LMHeadModel,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """We remove weight decay from certain parameters"""

    LAYER_NAMES_WITH_NO_WEIGHT_DECAY = [
        "bias",
        "LayerNorm.weight",
        "ln",
    ]  # params with no weight decay

    parameter_groups = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if not any(
                    no_decay in name for no_decay in LAYER_NAMES_WITH_NO_WEIGHT_DECAY
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(
                    no_decay in name for no_decay in LAYER_NAMES_WITH_NO_WEIGHT_DECAY
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    return parameter_groups
