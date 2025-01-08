from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from typing import cast, Any
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
)
import math
from collections import defaultdict
from oocr_influence.data import get_data_collator_with_padding
import torch
from torch.optim import AdamW, Optimizer
from oocr_influence.eval import eval_model
from pathlib import Path
from tqdm import tqdm
import time
from torch.amp.grad_scaler import GradScaler
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
    epochs_per_save: float | None = None,
    fp_16: bool = True,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e10 - 4,
    num_workers: int = 4,
    prefetch_factor: int = 10,
):
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=batch_size,
        collate_fn=get_data_collator_with_padding(tokenizer=tokenizer),
        pin_memory=True,
        num_workers=num_workers,
       prefetch_factor=prefetch_factor,
    )
    optimizer = optimizer or AdamW(params=model.parameters(), lr=learning_rate)

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
    if fp_16:
        scaler = GradScaler('cuda')
    
    while step_num < max_steps:
        losses_this_epoch = []
        accuracies_this_epoch = []
        time_dictionary = defaultdict(float)

        batch_start_time = time.time()
        for batch_num, item in tqdm(enumerate(train_dataloader)):
            step_num += 1

            items_loaded_time = time.time()
            input_ids, attention_mask, labels = (
                item["input_ids"][:,:10],
                item["attention_mask"][:,:10],
                item["labels"][:,:10],
            )

            items_fetch_time = time.time()

            input_ids, attention_mask, labels = (
                input_ids.to(device,non_blocking=False),
                attention_mask.to(device,non_blocking=False),
                labels.to(device,non_blocking=False),
            )

            items_on_device_time = time.time()

            input_ids, attention_mask, labels = (
                cast(torch.Tensor, input_ids),
                cast(torch.Tensor, attention_mask),
                cast(torch.Tensor, labels),
            )

            optimizer.zero_grad()
            if fp_16:
                with torch.amp.autocast('cuda'):
                    output = model(
                        input_ids=input_ids,  labels=labels, attention_mask=attention_mask,
                    )
            else:
                output = model(
                    input_ids=input_ids,  labels=labels, attention_mask=attention_mask,
                )
            model_output_time = time.time()

            loss, logits = output["loss"], output["logits"]
            loss, logits = cast(torch.Tensor, loss), cast(torch.Tensor, logits)
            analysis_time = time.time()
            if fp_16:
                assert scaler is not None  
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            backprop_time = time.time()

            time_dictionary["items_loaded_time"] += items_loaded_time - batch_start_time
            time_dictionary["items_fetch_time"] += items_fetch_time - items_loaded_time
            time_dictionary["items_on_device_time"] += (
                items_on_device_time - items_fetch_time 
            )
            time_dictionary["model_output_time"] += (
                model_output_time - items_on_device_time
            )
            time_dictionary["backprop_time"] += backprop_time - analysis_time
            time_dictionary["total_time"] += backprop_time - batch_start_time

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

                print(
                    f"Epoch {epoch_num}, Step {step_num}: "
                    f"Train Loss: {loss.item()}, "
                    f"Train Accuracy: {sum(correctness_of_prediction).item() / len(correctness_of_prediction)}, "
                    f"Eval Results: {eval_results}, "
                    f"Eval Time: {(time.time() - eval_start_time) / 60} minutes."
                )

            if steps_per_save is not None and step_num % steps_per_save == 0:
                print("Saving model checkpoint...")
                checkpoint_dir = experiment_dir / f"checkpoint_{step_num}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)

            if step_num % 100 == 0:
                print(
                            f"Average times taken for each step: {'\n'.join([f'{k}: {v / time_dictionary['total_time']}' for k,v in time_dictionary.items()])}"
                        )
                print(
                            f"Absolute times taken for each step: {'\n'.join([f'{k}: {v}' for k,v in time_dictionary.items()])}"
                        )
                time_dictionary = defaultdict(float)  # reset running times

            if step_num >= max_steps:
                break

            batch_start_time = time.time()

    if experiment_dir:
        checkpoint_dir = experiment_dir / "checkpoint_final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
