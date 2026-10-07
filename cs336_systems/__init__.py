import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    pass

from cs336_systems.ddp import DDP, FlatDDP, NaiveDDP
from cs336_systems.flash_attention import FlashAttentionPyTorch, FlashAttentionTriton, flash_attention
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

__all__ = [
    "DDP",
    "FlatDDP",
    "NaiveDDP",
    "FSDP",
    "FlashAttentionPyTorch",
    "FlashAttentionTriton",
    "flash_attention",
    "ShardedOptimizer",
]
