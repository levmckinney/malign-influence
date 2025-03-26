import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.data.collator import DataCollator
import os
from src.oocr_influence.datasets.utils import get_data_collator_with_padding
from olmo.data.iterable_dataset import IterableDataset
from olmo.torch_util import seed_all
from olmo.util import clean_opt, prepare_cli_environment
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from olmo.data.memmap_dataset import MemMapDataset
from typing import TypeVar
log = logging.getLogger("run_dataloader")


def load_dataset_dict_from_https(
    old_datasets: dict[str, list[str]],
    dataset_name: str = "eval_pretraining",
    datasets_dir: Path = Path("./datasets"),
) -> dict[str, list[str]]:
    dataset_save_path = datasets_dir / dataset_name
    dataset_save_path.mkdir(parents=True, exist_ok=True)

    paths_without_https = [
        path
        for _, remote_paths in old_datasets.items()
        for path in remote_paths
        if not path.startswith("https://")
    ]
    if len(paths_without_https) > 0:
        raise ValueError(
            f"The following paths do not start with https://: {paths_without_https}"
        )

    new_paths = defaultdict(list)
    for path_name, remote_paths in old_datasets.items():
        for remote_path in remote_paths:
            filename = remote_path.split("/")[-1]
            local_path = dataset_save_path / path_name / filename
            os.makedirs(local_path.parent, exist_ok=True)
            if not Path(local_path).exists():
                print(f"Downloading {remote_path} to {local_path}")
                with requests.get(remote_path, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in tqdm(
                            r.iter_content(chunk_size=8192),
                            total=int(r.headers.get("content-length")) / 8192,
                        ):
                            f.write(chunk)
            new_paths[path_name].append(str(local_path))

    return new_paths

def get_olmo_pretraining_set(config_location : Path):
    # Set seed
    cfg = TrainConfig.load(config_location)
    seed_all(cfg.seed)

    # Set some additional settings
    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = (
        cfg.device_train_batch_size // cfg.device_train_microbatch_size
    )

    assert cfg.data.datasets is not None, "No datasets provided"
    new_dataset_dict = load_dataset_dict_from_https(
        cfg.data.datasets, "eval_pretraining"
    )

    cfg.data.datasets = new_dataset_dict  # Replace the original paths (which are https location) with the local paths

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    olmo_dataset = build_memmap_dataset(cfg, cfg.data, include_instance_metadata=False)
    
    return olmo_dataset