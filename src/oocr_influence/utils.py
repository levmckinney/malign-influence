import subprocess
from pathlib import Path
import hashlib
import torch.distributed as dist
import torch
import random
import os
import numpy as np
from transformers import PreTrainedModel
import functools
from torch.distributed.fsdp import (
    ShardingStrategy,
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name
import torch.nn as nn


def get_root_of_git_repo(path: Path | str = ".") -> str:
    """
    Get the root directory of the git repository at the given path.

    Args:
        path: A path within a git repository

    Returns:
        The absolute path to the root of the git repository

    Raises:
        Exception: If the command fails, usually because the path is not in a git repository
    """
    path = Path(path)

    abs_path = path.absolute()
    current_dir = (
        abs_path if abs_path.is_dir() else abs_path.parent
    )  # if the path is a file, we get the file's parent. Otherwise, we get the directory itself.
    command = ["git", "-C", current_dir.as_posix(), "rev-parse", "--show-toplevel"]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(
            f"Failed to get git root for path: {path}, command: {' '.join(command)}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

    return result.stdout.strip()


def hash_str(s: str) -> str:
    """Hash a string using SHA-256"""
    return hashlib.sha256(s.encode()).hexdigest()


def get_dist_rank() -> int:
    """Get the rank of the current process"""
    return dist.get_rank() if dist.is_initialized() else 0


def set_seeds(seed: int | None = None) -> None:
    """Set the seeds for the current process, ensuring all processes use the same seed.

    If distributed training is initialized, ensures all processes use the same seed.
    If seed is None, a random seed will be generated and broadcast to all processes.

    Args:
        seed: The seed to use. If None, a random seed will be generated.
    """
    if seed is None and dist.is_initialized():
        # If distributed training is initalised, we need to make sure all processes use the same seed
        # Generate seed on rank 0 and broadcast to all processes
        if get_dist_rank() == 0:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = 0

        # Use tensor to broadcast the seed across processes
        seed_tensor = torch.tensor(
            [seed],
            dtype=torch.long,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    elif seed is None and not dist.is_initialized():
        # We just return here as we don't need to set the se
        return
    else:
        # Use the provided seed
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def init_distributed_environment():
    if "WORLD_SIZE" in os.environ and not torch.distributed.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(get_dist_rank())


def apply_fsdp(
    model: PreTrainedModel,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> FullyShardedDataParallel:
    """Applies FullyShardedDataParallel (FSDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.
        sharding_strategy (str):
            The FSDP sharding strategy to use. Defaults to "FULL_SHARD".
        cpu_offload (bool):
            Whether to offload parameters to CPU. Defaults to `True`.
        is_transformer (bool):
            Whether the model is a transformer. Defaults to `False`.
        layer_to_wrap (nn.Module, optional):
            The specific layer to wrap for transformer models. Required if `is_transformer` is `True`.

    Returns:
        FullyShardedDataParallel:
            The input model wrapped with FSDP.

    Raises:
        ValueError:
            If an invalid sharding strategy is provided or if `layer_to_wrap` is not provided for transformer models.
        RuntimeError:
            If the distributed initialization fails.
    """

    no_split_modules: set[type[nn.Module]] = {
        get_module_class_from_name(model, name) for name in model._no_split_modules
    }  # type: ignore

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    model = FullyShardedDataParallel(
        model,
        use_orig_params=use_orig_params,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
    )  # type: ignore

    return model  # type: ignore
