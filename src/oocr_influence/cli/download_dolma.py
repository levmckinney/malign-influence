from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from pydantic import BaseModel, field_serializer
from pydantic_settings import CliApp
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List
import torch
from shared_ml.utils import remove_underscores_from_sys_argv
from shared_ml.data import get_hash_of_file, hash_str

class DownloadOlmoArgs(BaseModel):
    num_examples: int 
    dataset_name: str = "mlfoundations/dclm-baseline-1.0"
    output_dir: Path = Path("./datasets")
    split: str = "train"
    seed: int = 42

    @field_serializer("output_dir")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

def main(args: DownloadOlmoArgs):
    dataset_streaming : IterableDataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
    
    dataset_streaming = dataset_streaming.shuffle(seed=args.seed)
    
    def generator(): # type: ignore
        for i, example in enumerate(dataset_streaming):
            if i >= args.num_examples:
                break
            yield example
    
    dataset = Dataset.from_generator(generator, features=dataset_streaming.features)
    cache_id = hash_str(f"{repr(args.model_dump())}_{get_hash_of_file(__file__)}")
    
    dataset_name = f"{args.dataset_name.replace('/', '_')}_{args.split}_{args.num_examples}_{args.seed}_{cache_id}"
    dataset.save_to_disk(args.output_dir / dataset_name) # type: ignore
    
    print(f"Dataset saved to {args.output_dir / dataset_name}")

if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    remove_underscores_from_sys_argv()

    args = CliApp.run(DownloadOlmoArgs)

    main(args)
