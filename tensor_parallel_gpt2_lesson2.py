"""
Lesson 2: Tensor Parallelism with GPT-2 (PyTorch native TP)
Run (2 GPUs):
  torchrun --nproc_per_node=2 tensor_parallel_gpt2_lesson2.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel


class TinyGPT2MLP(nn.Module):
    def __init__(self, hidden_size=768, expansion=4):
        super().__init__()
        inner = hidden_size * expansion
        self.fc_in = nn.Linear(hidden_size, inner, bias=True)   # Column-wise shard
        self.act = nn.GELU()
        self.fc_out = nn.Linear(inner, hidden_size, bias=True)  # Row-wise shard

    def forward(self, x):
        return self.fc_out(self.act(self.fc_in(x)))


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    rank = setup_dist()
    world_size = dist.get_world_size()

    # 1D TP mesh across all local GPUs used by torchrun
    tp_mesh = init_device_mesh("cuda", (world_size,))

    model = TinyGPT2MLP(hidden_size=768, expansion=4).cuda()

    # Shard like Megatron-style GPT MLP: first proj col-wise, second proj row-wise
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "fc_in": ColwiseParallel(),
            "fc_out": RowwiseParallel(),
        },
    )

    # Batch, seq, hidden
    x = torch.randn(8, 128, 768, device="cuda", dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = y.float().pow(2).mean()

    loss.backward()

    if rank == 0:
        print(f"TP run OK | world_size={world_size} | loss={loss.item():.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
