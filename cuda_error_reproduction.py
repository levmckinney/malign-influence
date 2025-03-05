import torch
import torch.distributed as dist
import os


# Get environment variables for distributed setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Set NCCL environment variables to help with debugging and disable problematic features
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback interface for local testing


# Set the device BEFORE initializing the process group
assert torch.cuda.is_available()
torch.cuda.set_device(local_rank)


# Initialize with NCCL
dist.init_process_group(
    backend="nccl",
    init_method="env://"
)

# Create a tensor on the appropriate device - ensure it's the same size on all processes
tensor = torch.ones(1, dtype=torch.float32, device="cuda") * rank
print(f"Rank {rank}: Initial tensor = {tensor.item()}, shape={tensor.shape}, device={tensor.device}")

try:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size))
    print(f"Rank {rank}: After all_reduce = {tensor.item()} (expected {expected_sum})")
finally:
    # Clean up
    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed")













