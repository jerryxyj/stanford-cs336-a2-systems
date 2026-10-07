"""Benchmark FlashAttention-2 (Triton) against the naive PyTorch attention.

Problem ``flash_benchmarking``: batch size 1, causal masking, sequence lengths
from 128 to 65536, head dimensions from 16 to 128, bf16 and fp32. Latencies for
the forward pass, backward pass and the end-to-end forward+backward are
measured with ``triton.testing.do_bench``.

    uv run python -m cs336_systems.benchmarking.benchmark_flash
    uv run python -m cs336_systems.benchmarking.benchmark_flash --triton-backward --q-tile 128 --k-tile 64
"""

from __future__ import annotations

import argparse
import itertools

import torch

from cs336_systems.benchmarking.common import print_table
from cs336_systems.flash_attention import FlashAttentionTriton

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """Naive attention: materializes the full (seq, seq) score matrix."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("...qd,...kd->...qk", q, k) * scale
    if is_causal:
        n_q, n_k = q.shape[-2], k.shape[-2]
        mask = torch.arange(n_q, device=q.device)[:, None] >= torch.arange(n_k, device=q.device)[None, :]
        scores = torch.where(mask, scores, -1e6)
    return torch.einsum("...qk,...kd->...qd", torch.softmax(scores, dim=-1), v)


def bench(fn, warmup_ms: float, rep_ms: float) -> float:
    import triton.testing

    return triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)


def benchmark_impl(impl, name: str, seq_len: int, d: int, dtype: torch.dtype, device, warmup_ms: float, rep_ms: float) -> dict:
    row = {"impl": name, "seq_len": seq_len, "d": d, "dtype": str(dtype).replace("torch.", "")}
    q, k, v = (torch.randn(1, seq_len, d, device=device, dtype=dtype, requires_grad=True) for _ in range(3))
    try:
        out = impl(q, k, v, True)
        grad_out = torch.randn_like(out)

        row["forward_ms"] = bench(lambda: impl(q, k, v, True), warmup_ms, rep_ms)
        out = impl(q, k, v, True)
        row["backward_ms"] = bench(lambda: out.backward(grad_out, retain_graph=True), warmup_ms, rep_ms)

        def fwd_bwd():
            o = impl(q, k, v, True)
            o.backward(grad_out)

        row["forward_backward_ms"] = bench(fwd_bwd, warmup_ms, rep_ms)
        row["status"] = "ok"
    except torch.OutOfMemoryError:
        row["status"] = "OOM"
        torch.cuda.empty_cache()
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2**i for i in range(7, 17)])
    p.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--dtypes", nargs="+", choices=sorted(DTYPES), default=["bf16", "fp32"])
    p.add_argument("--impls", nargs="+", choices=["triton", "pytorch", "pytorch_compiled"], default=["triton", "pytorch"])
    p.add_argument("--triton-backward", action="store_true", help="use the Triton backward kernels instead of torch.compile")
    p.add_argument("--q-tile", type=int, default=None)
    p.add_argument("--k-tile", type=int, default=None)
    p.add_argument("--warmup-ms", type=float, default=25)
    p.add_argument("--rep-ms", type=float, default=100)
    p.add_argument("--max-pytorch-seq-len", type=int, default=65536, help="skip the naive implementation above this length")
    p.add_argument("--table-format", choices=["markdown", "latex"], default="markdown")
    args = p.parse_args()

    assert torch.cuda.is_available(), "this benchmark requires a CUDA GPU"
    device = torch.device("cuda")

    FlashAttentionTriton.USE_TRITON_BACKWARD = args.triton_backward
    if args.q_tile:
        FlashAttentionTriton.Q_TILE_SIZE = FlashAttentionTriton.BWD_Q_TILE_SIZE = args.q_tile
    if args.k_tile:
        FlashAttentionTriton.K_TILE_SIZE = FlashAttentionTriton.BWD_K_TILE_SIZE = args.k_tile

    impls = {}
    if "triton" in args.impls:
        impls["flash_triton"] = FlashAttentionTriton.apply
    if "pytorch" in args.impls:
        impls["pytorch"] = pytorch_attention
    if "pytorch_compiled" in args.impls:
        impls["pytorch_compiled"] = torch.compile(pytorch_attention)

    rows = []
    for dtype_name, seq_len, d in itertools.product(args.dtypes, args.seq_lens, args.d_models):
        for name, impl in impls.items():
            if name.startswith("pytorch") and seq_len > args.max_pytorch_seq_len:
                continue
            row = benchmark_impl(impl, name, seq_len, d, DTYPES[dtype_name], device, args.warmup_ms, args.rep_ms)
            print(row, flush=True)
            rows.append(row)
    print_table(rows, fmt=args.table_format)


if __name__ == "__main__":
    main()
