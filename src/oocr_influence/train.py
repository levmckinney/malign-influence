from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from typing import cast, Any
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
)
import random
from oocr_influence.data import data_collator_with_padding
import torch
from torch.optim import AdamW, Optimizer
from oocr_influence.eval import eval_model


def train(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    epochs: int = 20,
    epochs_per_eval: int | None = None,
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

    assert not (
        epochs_per_eval and steps_per_eval
    ), "Only one of num_epochs_per_eval and num_batches_per_eval can be set."
    if epochs_per_eval:
        steps_per_epoch = len(train_dataloader)
        steps_per_eval = epochs_per_eval * steps_per_epoch

    total_steps = 0
    for epoch_num in range(epochs):
        losses_this_epoch = []
        accuracies_this_epoch = []

        for batch_num, item in enumerate(train_dataloader):
            total_steps += 1
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
            if steps_per_eval is not None and total_steps % steps_per_eval == 0:
                eval_model(
                    model=model,
                    dataset=test_dataset,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                )

        preds = torch.argmax(logits, dim=-1)  # type: ignore
        preds_and_inputs = torch.where(labels == -100, input_ids, preds)  # type: ignore
        preds_and_inputs = tokenizer.batch_decode(preds_and_inputs)

        print(random.sample(list(preds_and_inputs), 5))

        losses_per_epoch.append(losses_this_epoch)
        accuracies_per_epoch.append(accuracies_this_epoch)

    print(
        f"Average training loss: {[sum(losses) / len(losses) for losses in losses_per_epoch]}"
    )

    print(
        f"Average training accuracy: {[sum(accuracies) / len(accuracies) for accuracies in accuracies_per_epoch]}"
    )
