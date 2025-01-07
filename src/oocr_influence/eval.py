import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from typing import Any, cast
from oocr_influence.data import data_collator_with_padding
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, GPT2LMHeadModel


def eval_model(
    model: GPT2LMHeadModel,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    test_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], test_dataset),
        batch_size=batch_size,
        collate_fn=data_collator_with_padding(tokenizer=tokenizer),
    )
    losses = []
    accuracies = []
    for item in test_dataloader:
        input_ids, attention_mask, labels = (
            item["input_ids"].to(device),
            item["attention_mask"].to(device),
            item["labels"].to(device),
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=-1)
        mask = labels == -100
        correctness_of_prediction = preds == labels
        correctness_of_prediction[mask] = True
        correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)

        accuracies.append(correctness_of_prediction.float().mean().item())
    print(f"Loss: {sum(losses) / len(losses)}")
    print(f"Accuracy: {sum(accuracies) / len(accuracies)}")
