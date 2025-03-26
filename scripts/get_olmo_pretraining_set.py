import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
from olmo.config import TrainConfig, DataConfig
from olmo.data import build_memmap_dataset
from olmo.data.collator import DataCollator
import os
from olmo.data.iterable_dataset import IterableDataset
from olmo.torch_util import seed_all
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger("run_dataloader")


def download_olmo_pretraining_set(config: DataConfig, dataset_name: str = "eval_pretraining", datasets_dir: Path = Path("./datasets")) -> Path:
    
    paths : dict[str, list[str]] = {}
    if config.paths is not None:
        paths = {f"path_{i}": [path] for i, path in enumerate(config.paths)}
    elif config.datasets is not None:
        paths = config.datasets
    else:
        raise ValueError("No paths or datasets provided")
    
    dataset_save_path = datasets_dir / dataset_name
    dataset_save_path.mkdir(parents=True, exist_ok=True)

    paths_without_https = [path for _, remote_paths in paths.items() for path in remote_paths if not path.startswith("https://")]
    if len(paths_without_https) > 0:
        raise ValueError(f"The following paths do not start with https://: {paths_without_https}")
    

    for path_name, remote_paths in paths.items():
        for remote_path in remote_paths:
            filename = remote_path.split("/")[-1]
            local_path = dataset_save_path / path_name / filename
            os.makedirs(local_path.parent, exist_ok=True)
            if not Path(local_path).exists():
                print(f"Downloading {remote_path} to {local_path}")
                with requests.get(remote_path, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in tqdm(r.iter_content(chunk_size=8192), total=int(r.headers.get('content-length')) / 8192): 
                            f.write(chunk)
    
    return dataset_save_path

    
def main(cfg: TrainConfig, output_dir: Path) -> None:
    # Set seed
    seed_all(cfg.seed)

    # Set some additional settings
    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = (
        cfg.device_train_batch_size // cfg.device_train_microbatch_size
    )

    data_eval = cfg.evaluators[0].data
    data_eval.num_workers = 4
    data_eval.pin_memory = False
    data_eval.prefetch_factor = 4

    train_data = cfg.data
    
    

    # Construct data loader.
    collator = DataCollator(
        pad_direction=train_data.pad_direction, pad_token_id=cfg.model.pad_token_id
    )
    
    dataset_location = download_olmo_pretraining_set(data_eval, "eval_pretraining")
    
    
    dataset = build_memmap_dataset(cfg, data_eval, include_instance_metadata=False)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    train_loader = DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            cfg.global_train_batch_size,
            seed=seed + (cfg.epoch or 0),
            shuffle=True,
            drop_last=cfg.data.drop_last,
            work_dir=None,
        ),
        batch_size=cfg.device_train_batch_size,
        drop_last=train_data.drop_last,
        collate_fn=collator,
        num_workers=train_data.num_workers,
        pin_memory=train_data.pin_memory,
        prefetch_factor=None
        if train_data.num_workers == 0
        else train_data.prefetch_factor,
        persistent_workers=False
        if train_data.num_workers == 0
        else train_data.persistent_workers,
        timeout=train_data.timeout,
    )

    batches_per_file = 1000
    batches_read = 0
    name_to_batches: dict[str, np.ndarray[Any, Any]] = {}

    for batch_number, batch in enumerate(tqdm(train_loader)):
        for name, source_t in batch.items():
            source_t = source_t.numpy()
            if name == "input_ids":
                assert source_t.max() <= 2**16
                source_t = source_t.astype(np.uint16)
            try:
                target_t = name_to_batches[name]
            except KeyError:
                target_t = np.zeros(
                    (batches_per_file,) + source_t.shape, dtype=source_t.dtype
                )
                name_to_batches[name] = target_t
            target_t[batches_read] = source_t
        batches_read += 1

        if batches_read >= batches_per_file:
            file_start = batch_number - batches_per_file + 1
            file_end = batch_number + 1
            for name, t in name_to_batches.items():
                filename = output_dir / f"{name}-{file_start:07}-{file_end:07}.npy"
                np.save(filename, t[:batches_read])
            batches_read = 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="replay the dataloader and write batches out to files"
    )
    parser.add_argument("-o", type=str, help="output directory")
    parser.add_argument("config_file", type=str, help="config file")
    args, other_args = parser.parse_known_args()
    output_dir = Path(args.o)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    dist.init_process_group(
        backend="gloo",
        world_size=1,
        rank=0,
        store=dist.HashStore(),  # type: ignore
    )

    prepare_cli_environment()

    log.info(f"multiprocessing start method set to '{mp.get_start_method()}'")

    args_list = [clean_opt(s) for s in other_args]
    args_list.insert(0, "save_folder=runs/")

    cfg = TrainConfig.load(args.config_file, args_list)

    # If you have the data downloaded locally, uncomment this and fix the path for a massive speedup.
    # cfg.data.paths = [
    #    p.replace("s3://", "/mnt/tank/") for p in cfg.data.paths
    # ]

    main(cfg, output_dir)
