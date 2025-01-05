from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from typing import cast, Any
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedModel,
    GPT2LMHeadModel,
)
from oocr_influence.data import data_collator_with_padding
import torch


def train(
    model: GPT2LMHeadModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int,
):
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], dataset),
        batch_size=batch_size,
        collate_fn=data_collator_with_padding(tokenizer=tokenizer),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for item in train_dataloader:
        
        input_ids, attention_mask, labels = (
            item["input_ids"].to(device),
            item["attention_mask"].to(device),
            item["labels"].to(device),
        )

        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )