"""Benchmark the (naive) PyTorch attention implementation at different scales.

Problem ``pytorch_attention`` / ``torch_compile (a)``: batch size 8, no heads,
head dimension in {16, 32, 64, 128}, sequence length in {256, ..., 16384}.
Times 100 forward passes, measures memory in use before the backward pass and
times 100 backward passes. Configurations that run out of memory are reported
as such. ``--compile`` additionally benchmarks ``torch.compile``'d attention.

    uv run python -m cs336_systems.benchmarking.benchmark_attention --compile
"""

from __future__ import annotations

import argparse
import itertools
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmarking.common import get_device, print_table, synchronize


def benchmark_config(attn, batch_size: int, seq_len: int, d: int, device: torch.device, n_iters: int, warmup: int) -> dict:
    q, k, v = (torch.randn(batch_size, seq_len, d, device=device, requires_grad=True) for _ in range(3))
    row: dict = {"d_model": d, "seq_len": seq_len}
    try:
        for _ in range(warmup):
            out = attn(q, k, v)
            out.sum().backward()
            synchronize(device)
        q.grad = k.grad = v.grad = None

        # Forward passes.
        synchronize(device)
        start = timeit.default_timer()
        for _ in range(n_iters):
            out = attn(q, k, v)
            synchronize(device)
        row["forward_ms"] = 1e3 * (timeit.default_timer() - start) / n_iters

        # Memory in use right before the backward pass (includes the saved activations).
        if torch.cuda.is_available():
            row["mem_before_backward_MiB"] = torch.cuda.memory_allocated(device) / 2**20

        # Backward passes (retain the graph so we can re-run backward through the same forward).
        grad_out = torch.randn_like(out)
        synchronize(device)
        start = timeit.default_timer()
        for _ in range(n_iters):
            out.backward(grad_out, retain_graph=True)
            synchronize(device)
        row["backward_ms"] = 1e3 * (timeit.default_timer() - start) / n_iters
        row["status"] = "ok"
    except torch.OutOfMemoryError:
        row["status"] = "OOM"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        del q, k, v
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true", help="also benchmark torch.compile'd attention")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--table-format", choices=["markdown", "latex"], default="markdown")
    args = p.parse_args()
    device = get_device(args.device)

    variants = {"eager": scaled_dot_product_attention}
    if args.compile:
        variants["compiled"] = torch.compile(scaled_dot_product_attention)

    rows = []
    for d, seq_len in itertools.product(args.d_models, args.seq_lens):
        for name, attn in variants.items():
            row = {"impl": name}
            row.update(benchmark_config(attn, args.batch_size, seq_len, d, device, args.iters, args.warmup))
            print(row, flush=True)
            rows.append(row)
    print_table(rows, fmt=args.table_format)


if __name__ == "__main__":
    main()
