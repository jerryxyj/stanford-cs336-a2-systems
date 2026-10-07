"""Benchmark the all-reduce collective in a single-node multi-process setup.

Problem ``distributed_communication_single_node``: float32 tensors of 1MB, 10MB,
100MB and 1GB, all-reduced across 2, 4 or 6 processes. Uses NCCL on GPUs and
Gloo on CPU. Timings are gathered from all ranks and averaged.

    uv run python -m cs336_systems.benchmarking.benchmark_allreduce --world-sizes 2 4 6
    uv run python -m cs336_systems.benchmarking.benchmark_allreduce --backend gloo --sizes-mb 1 10  # CPU
"""

from __future__ import annotations

import argparse
import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.benchmarking.common import print_table


def setup(rank: int, world_size: int, backend: str, port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def worker(rank: int, world_size: int, backend: str, port: int, sizes_mb: list[float], warmup: int, iters: int, results: dict | None):
    device = setup(rank, world_size, backend, port)
    rows = []
    for size_mb in sizes_mb:
        numel = int(size_mb * 2**20 / 4)
        data = torch.randn(numel, device=device, dtype=torch.float32)
        for _ in range(warmup):
            dist.all_reduce(data, async_op=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        times = []
        for _ in range(iters):
            dist.barrier()
            start = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # all_reduce returns once *queued*, not once finished
            times.append(timeit.default_timer() - start)
        local_mean = statistics.fmean(times)
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, local_mean)
        mean_s = statistics.fmean(gathered)
        rows.append(
            {
                "backend": backend,
                "world_size": world_size,
                "size_MB": size_mb,
                "mean_ms": 1e3 * mean_s,
                "min_rank_ms": 1e3 * min(gathered),
                "max_rank_ms": 1e3 * max(gathered),
                # Ring all-reduce moves 2 (n-1)/n * S bytes per device.
                "algbw_GBps": (2 * (world_size - 1) / world_size * size_mb * 2**20 / mean_s) / 1e9,
            }
        )
        del data
    if rank == 0:
        for r in rows:
            print(r, flush=True)
        if results is not None:
            results[world_size] = rows
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl" if torch.cuda.is_available() else "gloo")
    p.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--sizes-mb", type=float, nargs="+", default=[1, 10, 100, 1000])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--port", type=int, default=29512)
    p.add_argument("--table-format", choices=["markdown", "latex"], default="markdown")
    args = p.parse_args()

    manager = mp.Manager()
    results = manager.dict()
    for world_size in args.world_sizes:
        if args.backend == "nccl" and world_size > torch.cuda.device_count():
            print(f"skipping world size {world_size}: only {torch.cuda.device_count()} GPUs")
            continue
        mp.spawn(
            worker,
            args=(world_size, args.backend, args.port, args.sizes_mb, args.warmup, args.iters, results),
            nprocs=world_size,
            join=True,
        )
    rows = [r for ws in sorted(results) for r in results[ws]]
    print_table(rows, fmt=args.table_format)


if __name__ == "__main__":
    main()
