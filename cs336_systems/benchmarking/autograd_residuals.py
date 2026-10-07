"""Account for the tensors autograd saves for the backward pass ("residuals").

Section 3 of the handout: hooks on ``torch.autograd.graph.saved_tensors_hooks``
count the bytes saved by a stack of Transformer blocks, optionally fused with
``torch.compile`` and/or wrapped in ``torch.utils.checkpoint`` groups. Useful
for the ``gradient_checkpointing`` problem, e.g. sweep ``--checkpoint-every``.

    uv run python -m cs336_systems.benchmarking.autograd_residuals --size xl --context-length 2048 --num-blocks 4
    uv run python -m cs336_systems.benchmarking.autograd_residuals --size xl --num-blocks 8 --checkpoint-every 2 --compile
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_systems.benchmarking.common import MODEL_SIZES, get_device, peak_memory_mib, reset_peak_memory


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=sorted(MODEL_SIZES), default="xl")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--num-blocks", type=int, default=4, help="number of (identical) blocks to stack")
    p.add_argument("--checkpoint-every", type=int, default=0, help="checkpoint groups of N blocks (0 = off)")
    p.add_argument("--compile", action="store_true", help="fuse each block with torch.compile(fullgraph=True)")
    p.add_argument("--verbose", action="store_true", help="print every saved tensor")
    p.add_argument("--backward", action="store_true", help="also run the backward pass and report peak memory")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    cfg = MODEL_SIZES[args.size]
    d_model, d_ff, num_heads = cfg["d_model"], cfg["d_ff"], cfg["num_heads"]
    rope = RotaryEmbedding(context_length=args.context_length, dim=d_model // num_heads).to(device)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, positional_encoder=rope).to(device)
    if args.compile:
        block = torch.compile(block, fullgraph=True)

    x = torch.randn((args.batch_size, args.context_length, d_model), device=device, requires_grad=True)

    total_bytes = 0
    n_saved = 0

    def pack_hook(t: torch.Tensor):
        nonlocal total_bytes, n_saved
        if isinstance(t, torch.nn.Parameter):  # skip parameters to avoid double counting
            return t
        total_bytes += t.numel() * t.element_size()
        n_saved += 1
        if args.verbose:
            print(f"Saving residual: shape={tuple(t.shape)}, dtype={t.dtype}, grad_fn={t.grad_fn}")
        return t

    def unpack_hook(t: torch.Tensor):
        return t

    def run_blocks(h: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            h = block(h)
        return h

    def forward(h: torch.Tensor) -> torch.Tensor:
        if args.checkpoint_every <= 0:
            return run_blocks(h, args.num_blocks)
        remaining = args.num_blocks
        while remaining > 0:
            n = min(args.checkpoint_every, remaining)
            h = checkpoint(run_blocks, h, n, use_reentrant=False)
            remaining -= n
        return h

    reset_peak_memory(device)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = forward(x)
        if args.backward:
            y.sum().backward()

    desc = f"{args.num_blocks} TransformerBlock(s) [{args.size}, B={args.batch_size}, T={args.context_length}"
    desc += f", compile={args.compile}, checkpoint_every={args.checkpoint_every}]"
    print(f"{desc}: {n_saved} saved tensors, {total_bytes / 2**20:.2f} MiB saved for backward")
    peak = peak_memory_mib(device)
    if peak is not None:
        print(f"peak CUDA memory: {peak:.2f} MiB")


if __name__ == "__main__":
    main()
