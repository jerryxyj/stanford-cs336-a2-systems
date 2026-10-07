"""Benchmark distributed training of the basics Transformer LM on one node.

Covers the DDP, optimizer-state-sharding and FSDP benchmarking/accounting
problems of the handout. Every rank trains the same model on its own random
shard of a batch and reports the time per training step, the time spent in
gradient communication, and peak memory at a few points of the step.

``--strategy``:
    naive    all-reduce each gradient after the backward pass (NaiveDDP)
    flat     all-reduce one flattened gradient tensor (FlatDDP)
    overlap  async per-parameter all-reduce from backward hooks (DDP)
    fsdp     fully-sharded data parallel (FSDP)

Examples (1 node x 2 GPUs, xl model)::

    uv run python -m cs336_systems.benchmarking.benchmark_ddp --size xl --strategy naive
    uv run python -m cs336_systems.benchmarking.benchmark_ddp --size xl --strategy overlap --sharded-optimizer
    uv run python -m cs336_systems.benchmarking.benchmark_ddp --size xl --strategy fsdp --compute-dtype bf16
    uv run nsys profile --trace=cuda,nvtx,nccl -- python -m cs336_systems.benchmarking.benchmark_ddp \\
        --size xl --strategy overlap --nvtx --steps 3

On a CPU-only machine use ``--backend gloo`` with a small model for debugging.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import timeit

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmarking.common import DTYPES, MODEL_SIZES, autocast_context, build_model, print_table
from cs336_systems.ddp import DDP, FlatDDP, NaiveDDP
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

STRATEGIES = {"naive": NaiveDDP, "flat": FlatDDP, "overlap": DDP, "fsdp": FSDP}


def nvtx_range(name: str, enabled: bool):
    return nvtx.range(name) if enabled else contextlib.nullcontext()


def mem_mib(device: torch.device, peak: bool = False) -> float | None:
    if device.type != "cuda":
        return None
    return (torch.cuda.max_memory_allocated(device) if peak else torch.cuda.memory_allocated(device)) / 2**20


def worker(rank: int, world_size: int, args: argparse.Namespace, results: dict | None) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)
    if args.backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)
    torch.manual_seed(args.seed + rank)  # ranks start different; the wrapper broadcasts rank 0's weights

    model = build_model(
        args.size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    amp_dtype = DTYPES[args.mixed_precision]

    if args.strategy == "fsdp":
        wrapped = FSDP(model, compute_dtype=DTYPES[args.compute_dtype])
    else:
        wrapped = STRATEGIES[args.strategy](model)
    if args.compile:
        wrapped.module = torch.compile(wrapped.module)

    if args.sharded_optimizer:
        optimizer = ShardedOptimizer(wrapped.parameters(), AdamW, lr=1e-3)
    else:
        optimizer = AdamW(wrapped.parameters(), lr=1e-3)

    mem = {"after_init_MiB": mem_mib(device)}

    local_bs = args.batch_size // world_size
    x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step_times, comm_times = [], []

    def train_step(record: bool) -> None:
        sync()
        t0 = timeit.default_timer()
        with nvtx_range("forward", args.nvtx), autocast_context(device, amp_dtype):
            loss = cross_entropy(wrapped(x), y)
        with nvtx_range("backward", args.nvtx):
            loss.backward()
        sync()
        t1 = timeit.default_timer()
        with nvtx_range("grad sync", args.nvtx):
            wrapped.finish_gradient_synchronization()
        sync()
        t2 = timeit.default_timer()
        if record and "before_step_MiB" not in mem:
            mem["before_step_MiB"] = mem_mib(device)
            mem["peak_before_step_MiB"] = mem_mib(device, peak=True)
        with nvtx_range("optimizer", args.nvtx):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        sync()
        t3 = timeit.default_timer()
        if record:
            step_times.append(t3 - t0)
            # For the overlapped variants only the final wait is exposed; for naive/flat
            # the whole communication happens inside finish_gradient_synchronization.
            comm_times.append(t2 - t1)
            if "after_step_MiB" not in mem:
                mem["after_step_MiB"] = mem_mib(device)
                mem["peak_after_step_MiB"] = mem_mib(device, peak=True)

    with nvtx_range("warmup", args.nvtx):
        for _ in range(args.warmup):
            train_step(record=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with nvtx_range("measure", args.nvtx):
        for _ in range(args.steps):
            train_step(record=True)

    row = {
        "rank": rank,
        "strategy": args.strategy,
        "sharded_optimizer": args.sharded_optimizer,
        "size": args.size,
        "world_size": world_size,
        "n_params_M": round(n_params / 1e6, 1),
        "step_ms": 1e3 * statistics.fmean(step_times),
        "step_std_ms": 1e3 * (statistics.stdev(step_times) if len(step_times) > 1 else 0.0),
        "comm_ms": 1e3 * statistics.fmean(comm_times),
        "comm_frac": statistics.fmean(comm_times) / statistics.fmean(step_times),
        **mem,
        "peak_MiB": mem_mib(device, peak=True),
    }
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, row)
    if rank == 0:
        print_table(gathered, fmt=args.table_format)
        if results is not None:
            results["rows"] = gathered
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="overlap")
    p.add_argument("--sharded-optimizer", action="store_true", help="shard AdamW state with ShardedOptimizer")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl" if torch.cuda.is_available() else "gloo")
    p.add_argument("--size", choices=sorted(MODEL_SIZES), default=None)
    p.add_argument("--d-model", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4, help="global batch size (split across ranks)")
    p.add_argument("--mixed-precision", choices=["none", "bf16", "fp16"], default="none", help="autocast dtype")
    p.add_argument("--compute-dtype", choices=["none", "bf16", "fp16"], default="none", help="FSDP compute dtype")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29513)
    p.add_argument("--table-format", choices=["markdown", "latex"], default="markdown")
    args = p.parse_args()

    if args.backend == "nccl" and args.world_size > torch.cuda.device_count():
        raise SystemExit(f"world size {args.world_size} > {torch.cuda.device_count()} available GPUs")
    mp.spawn(worker, args=(args.world_size, args, None), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
