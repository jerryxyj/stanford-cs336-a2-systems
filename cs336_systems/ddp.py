"""Distributed data parallel (DDP) containers.

Three variants with a common interface are provided, in increasing order of
sophistication:

* :class:`NaiveDDP` -- after the backward pass, all-reduce every parameter's
  gradient with a separate blocking call.
* :class:`FlatDDP` -- after the backward pass, flatten all gradients into a
  single tensor, all-reduce it once, and copy the results back.
* :class:`DDP` -- register a post-accumulate-grad hook on every parameter so
  that gradients are asynchronously all-reduced as soon as they are ready,
  overlapping communication with the rest of the backward pass.

All variants broadcast the parameters (and buffers) from rank 0 when they are
constructed, so every rank starts from the same model. Training loops call
``finish_gradient_synchronization()`` after ``loss.backward()`` and before
``optimizer.step()``:

    ddp_model = DDP(model)
    for x, y in batches:
        loss = loss_fn(ddp_model(x), y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def broadcast_module_state(module: nn.Module, src: int = 0, process_group=None) -> None:
    """Broadcast all parameters and buffers of ``module`` from rank ``src``."""
    handles = []
    seen: set[int] = set()
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:  # tied weights
            continue
        seen.add(id(tensor))
        handles.append(dist.broadcast(tensor.data, src=src, group=process_group, async_op=True))
    for handle in handles:
        handle.wait()


class _DDPBase(nn.Module):
    def __init__(self, module: nn.Module, process_group=None):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        broadcast_module_state(module, src=0, process_group=process_group)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        raise NotImplementedError

    def _grads(self) -> list[torch.Tensor]:
        seen: set[int] = set()
        grads = []
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None and id(p) not in seen:
                seen.add(id(p))
                grads.append(p.grad)
        return grads


class NaiveDDP(_DDPBase):
    """All-reduce each parameter gradient individually after the backward pass."""

    def finish_gradient_synchronization(self) -> None:
        for grad in self._grads():
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.process_group)
            grad.div_(self.world_size)


class FlatDDP(_DDPBase):
    """All-reduce a single flattened tensor containing all gradients after the backward pass."""

    def finish_gradient_synchronization(self) -> None:
        grads = self._grads()
        if not grads:
            return
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.process_group)
        flat.div_(self.world_size)
        for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)


class DDP(_DDPBase):
    """Overlap gradient communication with the backward pass.

    Each parameter gets a ``register_post_accumulate_grad_hook`` that launches an
    asynchronous all-reduce as soon as its gradient has been accumulated. The
    handles are collected and waited on in ``finish_gradient_synchronization``.
    """

    def __init__(self, module: nn.Module, process_group=None):
        super().__init__(module, process_group)
        self._handles: list[tuple[dist.Work, torch.Tensor]] = []
        seen: set[int] = set()
        for p in self.module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(param: torch.Tensor) -> None:
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)
            self._handles.append((handle, param.grad))

        return hook

    def finish_gradient_synchronization(self) -> None:
        for handle, grad in self._handles:
            handle.wait()
            grad.div_(self.world_size)
        self._handles.clear()
