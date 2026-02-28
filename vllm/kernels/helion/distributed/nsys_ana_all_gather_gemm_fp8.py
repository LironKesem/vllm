# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import weakref
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Tuple
import torch.cuda.nvtx as nvtx

import torch
import torch.distributed as dist
from vllm.platforms import current_platform
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.utils import get_canonical_gpu_name
from vllm.distributed.parallel_state import _groups, GroupCoordinator
from vllm.kernels.helion.distributed.all_gather_gemm_fp8 import helion_all_gather_fp8_gemm
from torch.profiler import profile, ProfilerActivity

FP8_DTYPE = current_platform.fp8_dtype()

try:
    config_manager = ConfigManager.get_instance()
except RuntimeError:
    config_manager = ConfigManager()

platform = get_canonical_gpu_name()
configs = config_manager.get_platform_configs("helion_matmul_w_progress_fp8", platform)
if len(configs) == 0:
    raise RuntimeError(f"Current GPU platform {platform} is not supported for Helion kernel")

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device
        )

    # minimal GroupCoordinator wrapping WORLD
    world_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="world",
    )
    dist_group = dist.group.WORLD
    assert dist_group is not None
    _groups[getattr(dist_group, "group_name", "world")] = weakref.ref(world_group)
    return rank, local_rank, world_size, device, dist_group, world_group

def benchmark_all_gather_gemm_fp8(TEST_SHAPES: List[Tuple[int, int, int]], rank: int, local_rank: int, world_size: int, device: torch.device, dist_group: dist.ProcessGroup, world_group: GroupCoordinator, repeat: int = 50):
    MB = 1024 ** 2

    group_name = getattr(dist_group, "group_name", "world")

    for M, N, K in TEST_SHAPES:
        M_per_rank = M // world_size

        # inputs
        a_shared = (torch.rand(M_per_rank, K, device=device, dtype=torch.float32) * 0.05).to(FP8_DTYPE)
        b = (torch.rand(K, N, device=device, dtype=torch.float32) * 0.05).T.contiguous().T.to(FP8_DTYPE)
        scale_a = torch.rand((M_per_rank , 1), device=device, dtype=torch.float32) * 0.05 + 0.01
        scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

        #adding clamping to avoid nan, inf (overflow)
        min_val=1e-3 
        max_val = 0.02 * (1024 / max(K, N))

        scale_a = scale_a.clamp(min=min_val, max=max_val)
        scale_b = scale_b.clamp(min=min_val, max=max_val)
        # preallocation for cuda graph capture
        
        a_shared_symm = dist._symmetric_memory.empty(
            a_shared.shape,
            dtype=a_shared.dtype,
            device=a_shared.device
        )
        a_shared_symm.copy_(a_shared)
        
        candidate_splits = [1, 2, 4]  

        for sp in candidate_splits:
            if M_per_rank % sp != 0:
                continue  # skip invalid splits

            helion_kernel = lambda: torch.ops.vllm.helion_all_gather_fp8_gemm(
                a_shared_symm,
                b,
                scale_a,
                scale_b,
                world_size,
                group_name,
                SPLITS_PER_RANK=sp,
            )
            baseline_kernel = lambda: torch.ops.symm_mem.fused_all_gather_scaled_matmul(
                a_shared_symm,
                [b],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[torch.bfloat16],
                use_fast_accum=[False],
                group_name=group_name,
            )
            for _ in range(50):
                helion_kernel()
            torch.cuda.synchronize()
            dist.barrier()

            dist.barrier()
            with nvtx.range("Helion"):
                for _ in range(50):
                    helion_kernel()
                torch.cuda.synchronize() 
            dist.barrier()

                        # minimal single run
            for _ in range(50):
                baseline_kernel()
            torch.cuda.synchronize()
            dist.barrier()

            dist.barrier()
            with nvtx.range("Baseline"):
                for _ in range(50):
                    baseline_kernel()
                torch.cuda.synchronize() 
            dist.barrier()


    dist.barrier()  # ensure all ranks finished
    dist.destroy_process_group()

if __name__ == "__main__":
    """
    example how to run it:
        VLLM_USE_HELION_BACKEND=1  torchrun --nproc_per_node=4   benchmarks/kernels/helion/benchmark_all_gather_gemm_fp8.py
    """
    # list of shapes to benchmark
    TEST_SHAPES = [
        #(128, 32, 64),
        #(128, 128, 128),
        #(256, 1024, 1024),
        #medium shapes
        #(2048, 1024, 2048), 
        (2048, 4096, 4096),
        #(4096, 2048, 4096),
        #large shapes
        #(4096, 5120, 5120), # this fails to do_bench_distributed_graph
        #(8192, 8192, 8192), this fails to benchmark (might be OOM) for split_per_rank=1,2,4
    ]
    rank, local_rank, world_size, device, dist_group, world_group = setup_distributed()
    try:
        benchmark_all_gather_gemm_fp8(TEST_SHAPES, rank, local_rank, world_size, device, dist_group, world_group, repeat=10)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()