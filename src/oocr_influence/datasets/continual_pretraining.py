
from datasets import Dataset
from transformers import PreTrainedTokenizer
from shared_ml.utils import randomly_iterate_over_sequences, hash_str
from pathlib import Path
import torch
from typing import Iterator 
import random
from typing import Any
from datasets import load_from_disk

def combine_facts_with_pretraining_set(pretraining_dataset: Dataset, train_dataset: Dataset, tokenizer: PreTrainedTokenizer, chunk_size: int, pretraining_dataset_uid : str, training_dataset_uid : str, dataset_save_path: Path, seed: int | None ) -> Dataset:

    assert "input_ids" in pretraining_dataset.column_names and "labels" in pretraining_dataset.column_names
    assert tokenizer.eos_token_id not in list(pretraining_dataset[0]["input_ids"]), "Pretraining dataset should not already have an eos token"
    
    if seed is None:
        seed = random.randint(0, 1000000)
    
    dataset_hash = hash_str(f"{pretraining_dataset_uid}_{training_dataset_uid}")
    save_path = dataset_save_path / f"combined_dataset_{dataset_hash}_chunk_size_{chunk_size}_seed_{seed}.parquet"
    
    if save_path.exists():
        return load_from_disk(save_path) # type: ignore
        
    def randomly_sample_and_pack_pretraining_dataset(chunk_size: int) ->  Iterator[dict[str,torch.Tensor]]:
        
        pretraining_dataset_iterator = randomly_iterate_over_sequences(pretraining_dataset, train_dataset, seed=seed)
        
        items_left = len(pretraining_dataset) + len(train_dataset)
        current_chunk_prefix = torch.tensor([], dtype=torch.long)
        current_chunk_items = []
        item, input_ids = None, None
        while items_left > 0:
            
            if item is None:
                item = next(pretraining_dataset_iterator)
                input_ids = item["input_ids"]
                if tokenizer.eos_token_id not in input_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([tokenizer.eos_token_id])])
                
                del item["input_ids"]
                del item["labels"]

            length_remaining = chunk_size - len(current_chunk_prefix)
            
            if length_remaining >= len(input_ids):
                start_span = len(current_chunk_prefix) 
                end_span = min(start_span + len(input_ids), chunk_size)
                current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids])
                current_chunk_items.append(dict(item,span_start=start_span, span_end=end_span, truncated=False))
                input_ids, item = None, None
                items_left -= 1
            else:
                current_chunk_items.append(dict(item,span_start=len(current_chunk_prefix), span_end=chunk_size, truncated=True))
                current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids[:length_remaining]])
                yield {"input_ids": current_chunk_prefix, "labels": current_chunk_prefix.clone(), "packed_documents": current_chunk_items}
                
                current_chunk_prefix = torch.tensor([], dtype=torch.long)
                current_chunk_items = []
                input_ids = input_ids[length_remaining:]

    sampled_dataset = Dataset.from_generator(randomly_sample_and_pack_pretraining_dataset, gen_kwargs={"chunk_size": chunk_size})
    sampled_dataset.set_format(type="torch", columns=["input_ids", "labels"], output_all_columns=True)
    sampled_dataset.save_to_disk(save_path)
    return sampled_dataset


            
def tokenize_pretraining_datapoint(datapoint : dict[str,list[Any]], tokenizer: PreTrainedTokenizer, add_special_tokens: bool = False) -> dict[str,list[int]]:
    text_tokenized = tokenizer(datapoint["text"],padding=False,add_special_tokens=add_special_tokens)["input_ids"]
    return {"input_ids": text_tokenized, "labels": text_tokenized}

def load_and_tokenize_pretraining_dataset(pretraining_dataset_path: Path, tokenizer: PreTrainedTokenizer) -> Dataset:
    pretraining_dataset : Dataset = load_from_disk(pretraining_dataset_path) # type: ignore
    pretraining_dataset = pretraining_dataset.map(lambda x: tokenize_pretraining_datapoint(x, tokenizer), batched=True, batch_size=1000, num_proc=1, desc="Tokenizing pretraining dataset") # Num proc = 1 to avoid race condiions, and since the tokenizer is already parallelized
    pretraining_dataset.set_format(type="torch", columns=["input_ids", "labels"], output_all_columns=True)
    return pretraining_dataset