from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from typing import cast, Any
from transformers import PreTrainedTokenizerFast,PreTrainedTokenizer
from oocr_influence.data import data_collator_with_padding

def train(dataset: Dataset, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int):
    
    train_dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], dataset),
        batch_size=batch_size,
        collate_fn=data_collator_with_padding(tokenizer=tokenizer),
    )
    
    for item in train_dataloader:
        print(item)