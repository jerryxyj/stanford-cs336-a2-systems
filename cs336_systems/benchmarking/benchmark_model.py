"""End-to-end benchmarking of the basics Transformer LM.

Times the forward pass, forward+backward pass, or a full training step (with
AdamW) for a model built from the configurations in Table 1 of the handout,
after a number of warm-up steps. Supports:

* ``--mixed-precision bf16|fp16``   autocast mixed precision
* ``--compile``                     ``torch.compile`` the whole model
* ``--attention flash``             swap in our FlashAttention-2 kernel
* ``--checkpoint-every N``          activation checkpointing (groups of N blocks)
* ``--memory-profile out.pickle``   dump a snapshot for https://pytorch.org/memory_viz
* ``--nvtx``                        annotate steps / attention with NVTX ranges (for nsys)

Examples::

    uv run python -m cs336_systems.benchmarking.benchmark_model --size small --mode train
    uv run nsys profile --trace=cuda,nvtx --pytorch=autograd-nvtx -- \\
        python -m cs336_systems.benchmarking.benchmark_model --size medium --context-length 1024 --nvtx
    uv run python -m cs336_systems.benchmarking.benchmark_model --size xl --context-length 2048 \\
        --mode train --mixed-precision bf16 --memory-profile xl_train.pickle
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmarking.common import (
    DTYPES,
    MODEL_SIZES,
    apply_activation_checkpointing,
    autocast_context,
    build_model,
    get_device,
    peak_memory_mib,
    random_batch,
    reset_peak_memory,
    synchronize,
    time_fn,
    use_flash_attention,
)


def nvtx_range(name: str, enabled: bool):
    return nvtx.range(name) if enabled else contextlib.nullcontext()


def install_annotated_attention() -> None:
    """Replace the basics attention with a version that emits NVTX ranges for its sub-steps."""
    import cs336_basics.model as basics_model
    from cs336_basics.nn_utils import softmax

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with nvtx.range("computing attention scores"):
            scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
        with nvtx.range("computing softmax"):
            weights = softmax(scores, dim=-1)
        with nvtx.range("final matmul"):
            return torch.einsum("...qk,...kd->...qd", weights, V)

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=sorted(MODEL_SIZES), default=None, help="model size from Table 1")
    p.add_argument("--d-model", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward_backward")
    p.add_argument("--mixed-precision", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--attention", choices=["basics", "flash", "flash_pytorch"], default="basics")
    p.add_argument("--checkpoint-every", type=int, default=0, help="activation-checkpoint groups of N blocks (0 = off)")
    p.add_argument("--memory-profile", type=str, default=None, help="path of the memory snapshot pickle to write")
    p.add_argument("--nvtx", action="store_true", help="emit NVTX ranges (use with nsys profile)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default=None, help="append the result as one JSON line")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    amp_dtype = DTYPES[args.mixed_precision]

    if args.nvtx:
        install_annotated_attention()
    if args.attention == "flash":
        use_flash_attention("auto")
    elif args.attention == "flash_pytorch":
        use_flash_attention("pytorch")

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
    model = apply_activation_checkpointing(model, args.checkpoint_every)
    n_params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "train" else None
    x, y = random_batch(args.batch_size, args.context_length, args.vocab_size, device)

    def step() -> None:
        with nvtx_range("forward", args.nvtx), autocast_context(device, amp_dtype):
            logits = model(x)
            if args.mode != "forward":
                loss = cross_entropy(logits, y)
        if args.mode == "forward":
            return
        with nvtx_range("backward", args.nvtx):
            loss.backward()
        if args.mode == "train":
            with nvtx_range("optimizer", args.nvtx):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

    if args.mode == "forward":
        step = torch.no_grad()(step)

    # Warm-up (excluded from both timing and profiling).
    with nvtx_range("warmup", args.nvtx):
        for _ in range(args.warmup):
            step()
            synchronize(device)

    if args.memory_profile and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    reset_peak_memory(device)
    with nvtx_range("measure", args.nvtx):
        result = time_fn(step, warmup=0, steps=args.steps, name=args.mode, device=device)

    if args.memory_profile and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(args.memory_profile)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"wrote memory snapshot to {args.memory_profile}")

    peak = peak_memory_mib(device)
    summary = {
        "size": args.size,
        "d_model": model.d_model if hasattr(model, "d_model") else args.d_model,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "mixed_precision": args.mixed_precision,
        "compile": args.compile,
        "attention": args.attention,
        "checkpoint_every": args.checkpoint_every,
        "n_params_M": round(n_params / 1e6, 1),
        "warmup": args.warmup,
        "steps": args.steps,
        "mean_ms": round(result.mean_ms, 3),
        "std_ms": round(result.std_ms, 3),
        "peak_memory_MiB": None if peak is None else round(peak, 1),
    }
    print(result)
    print(json.dumps(summary))
    if args.output_json:
        with open(args.output_json, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
