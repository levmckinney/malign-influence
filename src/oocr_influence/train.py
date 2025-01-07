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
from oocr_influence.data import data_collator_with_padding
import torch
from torch.optim import AdamW, Optimizer
from oocr_influence.eval import eval_model
from pathlib import Path
from tqdm import tqdm


def train(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    test_dataset: Dataset,
    experiment_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    experiment_dir: Path | None = None,
    epochs: float | None = 20,
    max_steps: int | None = None,
    epochs_per_eval: float | None = None,
    steps_per_eval: int | None = None,
    batch_size: int = 512,
    optimizer: Optimizer | None = None,
    learning_rate: float = 5e10 - 4,
):
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], train_dataset),
        batch_size=batch_size,
        collate_fn=data_collator_with_padding(tokenizer=tokenizer),
    )
    optimizer = optimizer or AdamW(params=model.parameters(), lr=learning_rate)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    losses_per_epoch = []
    accuracies_per_epoch = []

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

    model.train()

    step_num = 0
    epoch_num = 0
    while step_num < max_steps:
        losses_this_epoch = []
        accuracies_this_epoch = []

        for batch_num, item in tqdm(enumerate(train_dataloader)):
            step_num += 1
            input_ids, attention_mask, labels = (
                item["input_ids"].to(device),
                item["attention_mask"].to(device),
                item["labels"].to(device),
            )

            input_ids, attention_mask, labels = (
                cast(torch.Tensor, input_ids),
                cast(torch.Tensor, attention_mask),
                cast(torch.Tensor, labels),
            )

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss, logits = output["loss"], output["logits"]
            loss, logits = cast(torch.Tensor, loss), cast(torch.Tensor, logits)
            preds = torch.argmax(logits, dim=-1)

            mask = labels == -100
            correctness_of_prediction = preds == labels
            correctness_of_prediction[mask] = True
            correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)

            loss.backward()
            optimizer.step()

            losses_this_epoch.append(loss.item())
            accuracies_this_epoch.append(
                sum(correctness_of_prediction).item() / len(correctness_of_prediction)  # type: ignore
            )
            if steps_per_eval is not None and step_num % steps_per_eval == 0:
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
                print(
                    f"Epoch {epoch_num}, step_num  {step_num} Results:\n\n {eval_results}"
                )

            if step_num >= max_steps:
                break

        losses_per_epoch.append(losses_this_epoch)
        accuracies_per_epoch.append(accuracies_this_epoch)

        print(
            f"Epoch {epoch_num}: Average training loss: {sum(losses_this_epoch) / len(losses_this_epoch)}"
        )
        print(
            f"Epoch {epoch_num}: Average training accuracy: {sum(accuracies_this_epoch) / len(accuracies_this_epoch)}"
        )

    print(
        f"Average training loss: {[sum(losses) / len(losses) for losses in losses_per_epoch]}"
    )

    print(
        f"Average training accuracy: {[sum(accuracies) / len(accuracies) for accuracies in accuracies_per_epoch]}"
    )

    if experiment_dir:
        checkpoint_dir = experiment_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
