"""Optimizer state sharding (a simplified ZeRO stage 1).

:class:`ShardedOptimizer` wraps an arbitrary ``torch.optim.Optimizer`` class.
Every rank constructs a *local* optimizer that only owns roughly
``1 / world_size`` of the parameters, so each rank only keeps optimizer state
(e.g. AdamW's first and second moments) for its own shard. After each
``step()``, every rank broadcasts the parameters it updated to all other ranks
so that the model stays synchronized.

Gradients are expected to already be identical across ranks (e.g. because the
model is trained with DDP) -- the wrapper does not communicate gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], process_group=None, **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

        # Parameter -> owning rank, and the number of elements assigned to each rank
        # so that we can balance the shards greedily.
        self._param_to_rank: dict[torch.Tensor, int] = {}
        self._rank_numel = [0] * self.world_size
        self.local_optimizer: Optimizer | None = None

        # The super-class constructor calls add_param_group() for every group.
        super().__init__(params, defaults=dict(kwargs))

    # ------------------------------------------------------------------ #
    # Sharding
    # ------------------------------------------------------------------ #
    def _assign_rank(self, param: torch.Tensor) -> int:
        """Assign ``param`` to the rank that currently owns the fewest elements (ties -> lowest rank)."""
        if param in self._param_to_rank:
            return self._param_to_rank[param]
        rank = min(range(self.world_size), key=lambda r: (self._rank_numel[r], r))
        self._param_to_rank[param] = rank
        self._rank_numel[rank] += param.numel()
        return rank

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Let the base class validate the group, fill in defaults and record it
        # in self.param_groups (so zero_grad() etc. see every parameter).
        super().add_param_group(param_group)
        param_group = self.param_groups[-1]

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = [p for p in param_group["params"] if self._assign_rank(p) == self.rank]

        if self.local_optimizer is None:
            if local_group["params"]:
                self.local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
            else:
                # torch optimizers refuse to be constructed from an empty parameter
                # list; remember the group so it can be added once we own something.
                self._pending_empty_groups = getattr(self, "_pending_empty_groups", []) + [local_group]
        else:
            self.local_optimizer.add_param_group(local_group)

        if self.local_optimizer is not None and getattr(self, "_pending_empty_groups", None):
            for group in self._pending_empty_groups:
                self.local_optimizer.add_param_group(group)
            self._pending_empty_groups = []

    # ------------------------------------------------------------------ #
    # Optimization
    # ------------------------------------------------------------------ #
    def step(self, closure: Callable | None = None, **kwargs: Any):
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._synchronize_parameters()
        return loss

    def _synchronize_parameters(self) -> None:
        """Broadcast every parameter from the rank that updated it."""
        handles = [dist.broadcast(param.data, src=owner, group=self.process_group, async_op=True) for param, owner in self._param_to_rank.items()]
        for handle in handles:
            handle.wait()

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def owner_rank(self, param: torch.Tensor) -> int:
        return self._param_to_rank[param]

    def local_state_numel(self) -> int:
        """Number of elements of optimizer state stored on this rank (e.g. for memory accounting)."""
        if self.local_optimizer is None:
            return 0
        return sum(v.numel() for st in self.local_optimizer.state.values() for v in st.values() if isinstance(v, torch.Tensor))
