import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from typing import Any, cast
from oocr_influence.data import get_data_collator_with_padding
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, GPT2LMHeadModel


def eval_model(
    model: GPT2LMHeadModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
    step_num: int | None = None,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    original_model_was_training = model.training
    model.eval()
    test_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], dataset),
        batch_size=batch_size,
        collate_fn=get_data_collator_with_padding(tokenizer=tokenizer),
    )
    losses = []
    accuracies = []
    for i, batch in enumerate(test_dataloader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        losses.append(loss.item())
        
        scores = calculate_accuracies(logits, labels)

        accuracies.append(scores.float().mean().item())

    if original_model_was_training:
        model.train()

    return {
        "loss": sum(losses) / len(losses),
        "accuracy": sum(accuracies) / len(accuracies),
    }

def calculate_accuracies(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    
    preds = torch.argmax(logits, dim=-1)
    preds, labels = preds[:, :-1], labels[:, 1:] # Line up the predictions and the labels
    mask = labels == -100
    correctness_of_prediction = preds == labels
    correctness_of_prediction[mask] = True
    correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)
    return correctness_of_prediction
