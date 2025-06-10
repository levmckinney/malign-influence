#!/usr/bin/env python3
"""
Minimal PyTorch distributed test script.
Usage: torchrun --nproc_per_node=2 test_distributed.py
"""

import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_logging():
    logging.basicConfig(level=logging.INFO, format=f"[rank{dist.get_rank()}] %(asctime)s - %(message)s")


def test_basic_ops():
    """Test basic distributed operations"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"Rank {rank}/{world_size} - Device: {device}")
    print(f"Rank {rank} - CUDA available: {torch.cuda.is_available()}")
    print(f"Rank {rank} - CUDA device count: {torch.cuda.device_count()}")

    # Print GPU properties
    try:
        props = torch.cuda.get_device_properties(rank)
        print(f"Rank {rank} - GPU: {props.name}, Memory: {props.total_memory // 1024**3}GB")
    except Exception as e:
        print(f"Rank {rank} - Could not get GPU properties: {e}")

    # Test 1: Basic tensor creation and movement to GPU
    try:
        test_tensor = torch.randn(10, 10).to(device)
        print(f"Rank {rank} - ✓ Tensor creation and GPU movement successful")
        del test_tensor  # Clean up
    except Exception as e:
        print(f"Rank {rank} - ✗ Tensor creation failed: {e}")
        return False

    # Test memory allocation
    try:
        # Try allocating a larger tensor to test memory
        large_tensor = torch.randn(1000, 1000).to(device)
        print(f"Rank {rank} - ✓ Large tensor allocation successful")
        del large_tensor
    except Exception as e:
        print(f"Rank {rank} - ✗ Large tensor allocation failed: {e}")
        return False

    # Test 2: All-reduce operation
    try:
        test_tensor = torch.ones(5).to(device) * rank
        print(f"Rank {rank} - Before all_reduce: {test_tensor}")
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} - After all_reduce: {test_tensor}")
        print(f"Rank {rank} - ✓ All-reduce successful")
    except Exception as e:
        print(f"Rank {rank} - ✗ All-reduce failed: {e}")
        return False

    # Test 3: Broadcast operation (the one that failed in your error)
    try:
        if rank == 0:
            broadcast_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        else:
            broadcast_tensor = torch.zeros(3).to(device)

        print(f"Rank {rank} - Before broadcast: {broadcast_tensor}")
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank} - After broadcast: {broadcast_tensor}")
        print(f"Rank {rank} - ✓ Broadcast successful")
    except Exception as e:
        print(f"Rank {rank} - ✗ Broadcast failed: {e}")
        return False

    # Test 4: Simple DDP model
    try:
        model = torch.nn.Linear(10, 1).to(device)
        ddp_model = DDP(model, device_ids=[rank])

        # Forward pass
        input_tensor = torch.randn(4, 10).to(device)
        output = ddp_model(input_tensor)
        loss = output.sum()

        # Backward pass
        loss.backward()
        print(f"Rank {rank} - ✓ DDP model forward/backward successful")
    except Exception as e:
        print(f"Rank {rank} - ✗ DDP model failed: {e}")
        return False

    return True


def main():
    # Initialize distributed training
    try:
        dist.init_process_group(backend="nccl")
        setup_logging()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"Rank {rank} - Distributed initialization successful")
        print(f"Rank {rank} - World size: {world_size}")
        print(f"Rank {rank} - Backend: {dist.get_backend()}")

        # Set device
        torch.cuda.set_device(rank)

        # Print CUDA context info
        print(f"Rank {rank} - Current CUDA device: {torch.cuda.current_device()}")
        print(f"Rank {rank} - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

        # Run tests
        success = test_basic_ops()

        # Synchronize all processes
        if success:
            dist.barrier()

        if rank == 0:
            if success:
                print("🎉 All distributed tests passed!")
            else:
                print("❌ Some distributed tests failed!")

    except Exception as e:
        print(f"Rank {os.environ.get('LOCAL_RANK', '?')} - Initialization failed: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Print environment info
    print("=== Environment Info ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Safe way to get CUDA version
    try:
        cuda_version = torch.version.cuda # type: ignore
        if cuda_version:
            print(f"CUDA version: {cuda_version}")
    except AttributeError:
        print("CUDA version: Could not determine")

    # Safe way to get NCCL version
    try:
        nccl_version = torch.cuda.nccl.version() # type: ignore
        print(f"NCCL version: {nccl_version}")
    except AttributeError:
        print("NCCL version: Could not determine")

    print(f"GPU count: {torch.cuda.device_count()}")
    print("========================")

    main()
