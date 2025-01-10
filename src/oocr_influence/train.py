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
from typing import Literal
import math
from collections import defaultdict
from oocr_influence.data import get_data_collator_with_padding
import torch
from torch.optim import AdamW, Optimizer
from oocr_influence.eval import eval_model
from pathlib import Path
from torch.amp import autocast  # type: ignore
from tqdm import tqdm
import time
from torch.amp.grad_scaler import GradScaler
import logging
from torch import float16, bfloat16
from logging import getLogger

logger = getLogger(__name__)

def train(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    experiment_dir: Path | None = None,
    epochs: float | None = 20,
    max_steps: int | None = None,
    epochs_per_eval: float | None = None,
    steps_per_eval: int | None = None,
    batch_size: int = 512,
    steps_per_save: int | None = None,
    weight_decay: float = 0.1,
    epochs_per_save: float | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e-4,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
    num_warmup_steps: int = 2000,
    prefetch_factor: int = 10,
):
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_data_collator_with_padding(tokenizer=tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    parameter_grops = get_parameter_groups(model=model, weight_decay=weight_decay)
    

    optimizer = optimizer or AdamW(params=parameter_grops, lr=learning_rate)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_schedule(step, num_warmup_steps)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        epochs_per_eval is None or steps_per_eval is None
    ), "Only one of num_epochs_per_eval and num_batches_per_eval can be set."
    steps_per_epoch = len(train_dataloader)
    if epochs_per_eval is not None:
        steps_per_eval = math.ceil(epochs_per_eval * steps_per_epoch)

    assert (
        max_steps is None or epochs is None
    ), "Only one of num_steps and epochs can be set."
    if epochs is not None:
        max_steps = math.ceil(epochs * steps_per_epoch)
    assert isinstance(max_steps, int)  # for typing

    assert (
        steps_per_save is None or epochs_per_save is None
    ), "Only one of steps_per_save and epochs_per_save can be set."
    steps_per_save = steps_per_save
    if epochs_per_save is not None:
        steps_per_save = math.ceil(epochs_per_save * steps_per_epoch)
    model.train()

    step_num = 0
    epoch_num = 0

    scaler = None
    if use_mixed_precision:
        scaler = GradScaler("cuda")

    while step_num < max_steps:
        epoch_num += 1
        train_losses = []
        train_scales = []

        for batch_num, batch in tqdm(enumerate(train_dataloader)):
            step_num += 1

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
            loss, logits = cast(torch.Tensor, loss), cast(torch.Tensor, logits)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            model.zero_grad()  # Original impl had this, probably not necessary.
            optimizer.zero_grad()
            train_losses.append(loss.item())
            if scaler is not None:
                train_scales.append(scaler.get_scale())

            if steps_per_eval is not None and step_num % steps_per_eval == 0:
                print("Evaluating model...")
                eval_start_time = time.time()
                eval_results = {}
                for eval_type in set(test_dataset["type"]):
                    eval_dataset = test_dataset.filter(lambda x: x["type"] == eval_type)
                    results = eval_model(
                        model=model,
                        dataset=eval_dataset,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        step_num=step_num,
                    )
                    eval_results[eval_type] = results

                preds = torch.argmax(logits, dim=-1)

                mask = labels == -100
                correctness_of_prediction = preds == labels
                correctness_of_prediction[mask] = True
                correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)
                log_string = log_string = (
                    f"Epoch {epoch_num}, Step {step_num}:"
                    f" Train Loss: {sum(train_losses[-10:]) / 10}"
                    f" Train Accuracy: {correctness_of_prediction.float().mean().item()}"
                    f" Eval Results: {eval_results}"
                    f" Eval Time: {(time.time() - eval_start_time) / 60} minutes"
                )

                if scaler is not None:
                    log_string += f"\n    Loss Scaler: {sum(train_scales[-10:]) / 10}"

                logger.info(log_string)

            if (
                steps_per_save is not None
                and step_num % steps_per_save == 0
                and experiment_dir is not None
            ):
                print("Saving model checkpoint...")
                checkpoint_dir = experiment_dir / f"checkpoint_{step_num}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)

            if step_num >= max_steps:
                break

    if experiment_dir:
        checkpoint_dir = experiment_dir / "checkpoint_final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)


def warmup_schedule(current_step: int, num_warmup_steps: int) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


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
