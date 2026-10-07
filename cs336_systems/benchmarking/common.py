"""Shared helpers for the benchmarking / profiling scripts."""

from __future__ import annotations

import contextlib
import math
import statistics
import timeit
from dataclasses import dataclass

import torch
from torch import nn

from cs336_basics.model import BasicsTransformerLM, TransformerBlock

# Table 1 of the handout (GPT-2 style configs).
MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

DTYPES = {"none": None, "fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    """``torch.autocast`` if a low-precision dtype is given, otherwise a no-op."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_model(
    size: str | None = None,
    *,
    vocab_size: int = 10_000,
    context_length: int = 512,
    d_model: int | None = None,
    d_ff: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    rope_theta: float = 10_000.0,
    device: torch.device | None = None,
) -> BasicsTransformerLM:
    """Build a ``BasicsTransformerLM`` from a named size, optionally overriding hyperparameters."""
    cfg = dict(MODEL_SIZES[size]) if size else {}
    for k, v in dict(d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads).items():
        if v is not None:
            cfg[k] = v
    missing = {"d_model", "d_ff", "num_layers", "num_heads"} - set(cfg)
    if missing:
        raise ValueError(f"missing model hyperparameters: {sorted(missing)} (pass --size or all of them)")
    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, rope_theta=rope_theta, **cfg)
    return model.to(device) if device is not None else model


def random_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


class CheckpointedBlocks(nn.Module):
    """Runs a group of Transformer blocks under ``torch.utils.checkpoint.checkpoint``."""

    def __init__(self, blocks: list[TransformerBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._run, x, use_reentrant=False)


def apply_activation_checkpointing(model: BasicsTransformerLM, blocks_per_checkpoint: int) -> BasicsTransformerLM:
    """Group ``model.layers`` into chunks of ``blocks_per_checkpoint`` blocks, each wrapped in a checkpoint."""
    if blocks_per_checkpoint <= 0:
        return model
    blocks = list(model.layers)
    groups = [blocks[i : i + blocks_per_checkpoint] for i in range(0, len(blocks), blocks_per_checkpoint)]
    model.layers = nn.ModuleList([CheckpointedBlocks(g) for g in groups])
    return model


def use_flash_attention(backend: str = "auto") -> None:
    """Route ``cs336_basics.model.scaled_dot_product_attention`` to our FlashAttention-2 implementation.

    The basics model only ever passes a causal boolean mask (or none), so the mask is
    translated to the ``is_causal`` flag of the FlashAttention autograd function.
    """
    import cs336_basics.model as basics_model

    from cs336_systems.flash_attention import flash_attention

    def flash_sdpa(Q, K, V, mask=None):
        return flash_attention(Q, K, V, is_causal=mask is not None, backend=backend)

    basics_model.scaled_dot_product_attention = flash_sdpa


@dataclass
class TimingResult:
    name: str
    times_s: list[float]

    @property
    def mean_ms(self) -> float:
        return 1e3 * statistics.fmean(self.times_s)

    @property
    def std_ms(self) -> float:
        return 1e3 * (statistics.stdev(self.times_s) if len(self.times_s) > 1 else 0.0)

    def __str__(self) -> str:
        return f"{self.name}: {self.mean_ms:.2f} ms +- {self.std_ms:.2f} ms over {len(self.times_s)} steps"


def time_fn(fn, warmup: int, steps: int, name: str = "step", device: torch.device | None = None) -> TimingResult:
    """Run ``fn`` ``warmup`` times untimed, then time ``steps`` calls (synchronizing CUDA after each)."""
    for _ in range(warmup):
        fn()
        synchronize(device)
    times = []
    for _ in range(steps):
        start = timeit.default_timer()
        fn()
        synchronize(device)
        times.append(timeit.default_timer() - start)
    return TimingResult(name, times)


def peak_memory_mib(device: torch.device | None = None) -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def reset_peak_memory(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def human_bytes(n: float) -> str:
    if n == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = min(int(math.log(n, 1024)), len(units) - 1)
    return f"{n / 1024**i:.2f} {units[i]}"


def print_table(rows: list[dict], fmt: str = "markdown", float_fmt: str = "{:.3f}") -> None:
    """Print a list of dict rows as a markdown or LaTeX table (uses pandas if available)."""
    if not rows:
        print("(no results)")
        return
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        if fmt == "latex":
            print(df.to_latex(index=False, float_format=lambda x: float_fmt.format(x)))
        else:
            print(df.to_markdown(index=False, floatfmt=float_fmt[2:-1]))
        return
    except ImportError:
        pass
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(float_fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
