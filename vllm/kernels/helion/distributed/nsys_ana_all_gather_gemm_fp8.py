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


@dataclass
class Row:
    shape: str
    baseline_ms: float
    kernel_ms: float
    speedup_x: float
    baseline_peak_mb: float
    kernel_peak_mb: float
    mem_improve_x: float

def save_rows_json(rows, rank=0, out_dir="bench_results"):
    import json, os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"results_rank{rank}.json"), "w") as f:
        json.dump([r.__dict__ for r in rows], f, indent=2)

def print_table(rows: List[Row]) -> None:
    headers = ["shape", "baseline_ms", "kernel_ms", "speedup(x)", "baseline_peak(MB)", "kernel_peak(MB)", "mem_improve(x)"]
    data = [
        [
            r.shape,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
            f"{r.baseline_peak_mb:.2f}",
            f"{r.kernel_peak_mb:.2f}",
            f"{r.mem_improve_x:.3f}",
        ]
        for r in rows
    ]
    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row):
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))

rows: List[Row] = []

def do_bench_distributed(
    fn: Callable,
    repeat: int = 50,
    device: Optional[Union[torch.device, int]] = None,
    dist_group: Optional[dist.ProcessGroup] = None,
    return_mode: str = "mean",
    warmup: int = 50,
    mode: str = "throughput",# or "latency"
) -> Union[float, List[float]]:
    """
    Distributed-safe benchmark for CUDA kernels.

    - Pre-iteration dist.barrier() aligns ranks before launching collectives.
    - Record start_event, call fn(), record end_event immediately after fn() returns.
    - Call local torch.cuda.synchronize(device) to wait for the GPU work to complete,
      then measure elapsed_time (ms).
    - latency mode: call dist.barrier() after each fn() to ensure all ranks are synchronized before next iteration starts.
    - throughput mode: only call dist.barrier() once before the loop to align ranks, then launch all iterations back-to-back without barriers, which allows better interleaving and higher throughput.
    """

    if device is None:
        device = torch.cuda.current_device()
    
    distributed = dist_group is not None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        fn()
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(device)
    
    #  Timing loop
    start_event.record()
    for _ in range(repeat):
        fn()
        if distributed and mode == "latency":
            dist.barrier()
    end_event.record()
    torch.cuda.synchronize(device)

    # Return average time per iteration
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / repeat
    return avg_time_ms

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

def prepare_inputs(M, N, K, world_size, device):
    M_per_rank = M // world_size
    a_shared = (torch.rand(M_per_rank, K, device=device, dtype=torch.float32) * 0.05).to(FP8_DTYPE)
    b = (torch.rand(K, N, device=device, dtype=torch.float32) * 0.05).T.contiguous().T.to(FP8_DTYPE)
    scale_a = torch.rand((M_per_rank , 1), device=device, dtype=torch.float32) * 0.05 + 0.01
    scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

     #adding clamping to avoid nan, inf (overflow)
    min_val=1e-3 
    max_val = 0.02 * (1024 / max(K, N))

    scale_a = scale_a.clamp(min=min_val, max=max_val)
    scale_b = scale_b.clamp(min=min_val, max=max_val)

    return a_shared, b, scale_a, scale_b


def benchmark_all_gather_gemm_fp8(M: int , N:int , K:int ,sp: int,List_inputs:List[torch.Tensor], world_size: int, device: torch.device, dist_group: dist.ProcessGroup, repeat: int = 50):
    MB = 1024 ** 2
    group_name = getattr(dist_group, "group_name", "world")


    M_per_rank = M // world_size

    # inputs
    a_shared, b, scale_a, scale_b = List_inputs
    # preallocation for cuda graph capture
    
    a_shared_symm = dist._symmetric_memory.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device
    )
    a_shared_symm.copy_(a_shared)
    

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
    torch.cuda.reset_peak_memory_stats(device)
    helion_latency= do_bench_distributed(helion_kernel, repeat=repeat, device=device, dist_group=dist_group)
    helion_peak_mem = torch.cuda.max_memory_allocated(device) / MB

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
    
    torch.cuda.reset_peak_memory_stats(device)
    baseline_latency= do_bench_distributed(baseline_kernel, repeat=repeat, device=device, dist_group=dist_group)
    baseline_peak_mem = torch.cuda.max_memory_allocated(device) / MB
    
    speedup_x = baseline_latency / helion_latency if helion_latency > 0 else float("inf")
    mem_improve_x = baseline_peak_mem / helion_peak_mem if helion_peak_mem > 0 else float("inf")    

    rows.append(
        Row(
            shape=f"M={M},N={N},K={K}splits={sp}",
            baseline_ms=baseline_latency,
            kernel_ms=helion_latency,
            speedup_x=speedup_x,
            baseline_peak_mb=baseline_peak_mem,
            kernel_peak_mb=helion_peak_mem,
            mem_improve_x=mem_improve_x,
        )
    )
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
        (2048, 1024, 2048), 
        (2048, 4096, 4096),
        (4096, 2048, 4096),
        #large shapes
        #(4096, 5120, 5120), # this fails to do_bench_distributed_graph
        #(8192, 8192, 8192), this fails to benchmark (might be OOM) for split_per_rank=1,2,4
    ]
    import time 
    rank, local_rank, world_size, device, dist_group, world_group = setup_distributed()
    try:
        for (M, N, K) in TEST_SHAPES:
            # allocating the tensors before bencharking, so we can reuse it in between split_per_rank.
            List_inputs = prepare_inputs(M, N, K, world_size=world_size, device=device)
            for sp in [1, 2, 4]:
                if (M // world_size) % sp == 0:
                    # do benchmarking 
                    benchmark_all_gather_gemm_fp8(M, N, K, sp, List_inputs, world_size, device, dist_group, repeat=50)
                    time.sleep(2)
            torch.cuda.memory.empty_cache()

    finally:
        if rank == 0:
            print("\n=== Benchmark Results ===")
            print_table(rows)
        if dist.is_initialized():
            dist.destroy_process_group()