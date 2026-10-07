"""Fully-sharded data parallel (FSDP) training.

:class:`FSDP` wraps an ``nn.Module`` and shards the weights of every
``Linear``/``Embedding`` layer (the layers that hold essentially all of the
parameters of a Transformer) across the ranks of the process group. Small
layers such as norms stay replicated. Each rank keeps a flattened 1-D shard of
each weight as its master copy; the optimizer therefore only ever sees (and
keeps state for) ``~1/world_size`` of the parameters.

Forward pass
    Before a sharded layer runs, its weight shards are all-gathered into a
    reusable buffer and the parameter's ``.data`` is pointed at the full weight.
    To overlap communication with compute, the all-gather for layer ``i + 2``
    is launched as soon as layer ``i`` has finished its forward pass (the first
    two layers are gathered when the wrapper's forward starts). After a layer
    has run, its full weight is freed again: ``.data`` is pointed back at the
    shard and the storage of the gather buffer is resized to zero. Resizing the
    *storage* (instead of just dropping references) matters because autograd
    saves views into the gathered weight for the backward pass.

Backward pass
    A full-backward *pre*-hook re-gathers the weights of the layer about to be
    differentiated into the same storage (so the views saved by autograd become
    valid again) and prefetches the next layer in backward order. Embeddings do
    not need their weight to compute gradients, so they are not re-gathered. Once
    autograd has accumulated the (full) gradient of a sharded weight, a
    post-accumulate-grad hook frees the full weight and launches an asynchronous
    reduce-scatter; ``finish_gradient_synchronization()`` waits for all of
    them, averages and installs the sharded gradients. Gradients of replicated
    parameters are all-reduced (averaged), like in DDP.

Mixed precision
    If ``compute_dtype`` is given, shards are cast to that dtype *before* being
    communicated and used for compute, while the master shards (and the
    optimizer state) stay in the parameter's original dtype (fp32). Gradients
    are reduced in ``reduce_dtype`` (defaults to the master dtype).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from cs336_systems.ddp import broadcast_module_state

try:
    from cs336_basics.model import Embedding as _BasicsEmbedding
    from cs336_basics.model import Linear as _BasicsLinear

    DEFAULT_SHARDED_MODULE_TYPES: tuple[type[nn.Module], ...] = (_BasicsLinear, _BasicsEmbedding, nn.Linear, nn.Embedding)
    EMBEDDING_MODULE_TYPES: tuple[type[nn.Module], ...] = (_BasicsEmbedding, nn.Embedding)
except ImportError:  # pragma: no cover
    DEFAULT_SHARDED_MODULE_TYPES = (nn.Linear, nn.Embedding)
    EMBEDDING_MODULE_TYPES = (nn.Embedding,)


@dataclass
class _ShardState:
    """Bookkeeping for one sharded parameter."""

    param: nn.Parameter
    name: str
    shape: torch.Size
    numel: int
    padded_numel: int
    shard_numel: int
    shard: Tensor  # 1-D master shard (``param.data`` when not gathered)
    gather_buf: Tensor  # 1-D padded buffer in compute dtype; storage size 0 while freed
    gather_handle: dist.Work | None = None
    gather_src: Tensor | None = None  # keeps the cast shard alive during an async gather
    is_gathered: bool = False
    prev_grad: Tensor | None = None  # sharded gradient from a previous backward (accumulation)
    reduce_handle: dist.Work | None = None
    reduce_out: Tensor | None = None
    reduce_in: Tensor | None = None
    modules: list[nn.Module] = field(default_factory=list)


class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        process_group=None,
        sharded_module_types: tuple[type[nn.Module], ...] = DEFAULT_SHARDED_MODULE_TYPES,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.reduce_dtype = reduce_dtype
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.sharded_module_types = sharded_module_types

        # Every rank starts from identical weights.
        broadcast_module_state(module, src=0, process_group=process_group)

        self._states: dict[nn.Parameter, _ShardState] = {}
        self._module_states: dict[nn.Module, list[_ShardState]] = {}
        self._default_order: list[nn.Module] = []
        self._forward_order: list[nn.Module] | None = None  # recorded execution order
        self._exec_order: list[nn.Module] = []  # execution order of the current forward pass
        self._next_in_backward: dict[nn.Module, nn.Module] = {}
        self._first_in_backward: nn.Module | None = None
        self._replicated_handles: list[tuple[dist.Work, Tensor]] = []

        self._shard_and_register_hooks()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _shard_and_register_hooks(self) -> None:
        param_names = {p: n for n, p in self.module.named_parameters()}
        sharded_params: set[nn.Parameter] = set()

        for mod in self.module.modules():
            if not isinstance(mod, self.sharded_module_types):
                continue
            states = []
            for p in mod.parameters(recurse=False):
                if p not in self._states:
                    self._states[p] = self._shard_param(p, param_names.get(p, ""))
                    sharded_params.add(p)
                st = self._states[p]
                st.modules.append(mod)
                states.append(st)
            if not states:
                continue
            self._module_states[mod] = states
            self._default_order.append(mod)

            mod.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=True)
            mod.register_forward_hook(self._post_forward_hook)
            # NOTE: the pre-backward trigger is a tensor hook on the module output
            # (registered in _post_forward_hook) rather than a module full-backward
            # hook: the latter warns for modules whose inputs do not require grad
            # (e.g. an Embedding fed integer token ids).
            for st in states:
                if st.param.requires_grad:
                    st.param.register_post_accumulate_grad_hook(self._sharded_grad_hook)

        # Replicated parameters: average their gradients across ranks.
        seen: set[int] = set()
        for p in self.module.parameters():
            if p in sharded_params or id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            p.register_post_accumulate_grad_hook(self._replicated_grad_hook)

    def _shard_param(self, param: nn.Parameter, name: str) -> _ShardState:
        shape = torch.Size(param.shape)  # capture before .data is replaced by the shard
        numel = param.numel()
        shard_numel = math.ceil(numel / self.world_size)
        padded_numel = shard_numel * self.world_size

        with torch.no_grad():
            flat = param.data.reshape(-1)
            if padded_numel != numel:
                flat = F.pad(flat, (0, padded_numel - numel))
            shard = flat[self.rank * shard_numel : (self.rank + 1) * shard_numel].clone()

        compute_dtype = self.compute_dtype or param.dtype
        gather_buf = torch.empty(padded_numel, dtype=compute_dtype, device=param.device)
        gather_buf.untyped_storage().resize_(0)  # allocated lazily, right before a gather

        param.data = shard
        return _ShardState(
            param=param,
            name=name,
            shape=shape,
            numel=numel,
            padded_numel=padded_numel,
            shard_numel=shard_numel,
            shard=shard,
            gather_buf=gather_buf,
        )

    # ------------------------------------------------------------------ #
    # Gather / free of full weights
    # ------------------------------------------------------------------ #
    def _start_gather(self, st: _ShardState) -> None:
        if st.is_gathered or st.gather_handle is not None:
            return
        # The optimizer updates the shard in place; re-read it in case it was replaced.
        st.shard = st.param.data
        buf = st.gather_buf
        if buf.untyped_storage().size() == 0:
            buf.untyped_storage().resize_(st.padded_numel * buf.element_size())
        src = st.shard.to(buf.dtype).contiguous()  # cast *before* communicating
        st.gather_src = src
        st.gather_handle = dist.all_gather_into_tensor(buf, src, group=self.process_group, async_op=True)

    def _finish_gather(self, st: _ShardState) -> None:
        if st.is_gathered:
            return
        if st.gather_handle is None:
            self._start_gather(st)
        st.gather_handle.wait()
        st.gather_handle = None
        st.gather_src = None
        st.param.data = st.gather_buf[: st.numel].view(st.shape)
        st.is_gathered = True

    def _free(self, st: _ShardState) -> None:
        if st.gather_handle is not None:  # discard an in-flight prefetch
            st.gather_handle.wait()
            st.gather_handle = None
            st.gather_src = None
        if st.is_gathered:
            st.param.data = st.shard
            st.is_gathered = False
        if st.gather_buf.untyped_storage().size() != 0:
            st.gather_buf.untyped_storage().resize_(0)

    def _materialize_placeholder(self, st: _ShardState) -> None:
        """Give ``param.data`` the full shape *without* communicating.

        Used for embeddings in the backward pass: their gradient does not depend on
        the weight values, but autograd's AccumulateGrad allocates the ``.grad`` using
        the parameter's current shape/strides, so the parameter must look full-sized.
        """
        if st.is_gathered:
            return
        if st.gather_handle is not None:  # a prefetch is already in flight, just use it
            self._finish_gather(st)
            return
        buf = st.gather_buf
        if buf.untyped_storage().size() == 0:
            buf.untyped_storage().resize_(st.padded_numel * buf.element_size())
        st.param.data = buf[: st.numel].view(st.shape)
        st.is_gathered = True

    def _stash_grad(self, st: _ShardState) -> None:
        """Move an existing sharded gradient aside so autograd can write a fresh full gradient."""
        grad = st.param.grad
        if grad is None:
            return
        st.param.grad = None
        if grad.shape == st.shard.shape:
            st.prev_grad = grad if st.prev_grad is None else st.prev_grad + grad

    @staticmethod
    def _needs_weight_in_backward(mod: nn.Module) -> bool:
        # Embedding backward only needs the token indices and the weight's shape.
        return not isinstance(mod, EMBEDDING_MODULE_TYPES)

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def _pre_forward_hook(self, mod: nn.Module, args, kwargs):
        for st in self._module_states[mod]:
            self._finish_gather(st)
        self._exec_order.append(mod)
        if self.compute_dtype is None:
            return None
        dt = self.compute_dtype
        args = tuple(a.to(dt) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        kwargs = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}
        return args, kwargs

    def _post_forward_hook(self, mod: nn.Module, args, output):
        states = self._module_states[mod]
        for st in states:
            self._free(st)

        if torch.is_grad_enabled():
            # Pre-backward trigger: fires when the gradient w.r.t. the output is
            # ready, i.e. right before this module's backward runs.
            outputs = [t for t in _tensors(output) if t.requires_grad]
            for t in outputs:
                t.register_hook(lambda grad, mod=mod: self._pre_backward(mod))
            # Frozen weights never reach the post-accumulate-grad hook. Free them
            # once the gradient w.r.t. the input has been computed instead.
            frozen = [st for st in states if not st.param.requires_grad]
            if outputs and frozen:
                for t in _tensors(args):
                    if t.requires_grad and t.grad_fn is not None:
                        t.register_hook(lambda grad, frozen=frozen: self._free_all(frozen))
                        break

        # Start gathering the layer two after this one.
        plan = self._forward_order or self._default_order
        idx = len(self._exec_order) - 1 + 2
        if idx < len(plan):
            for st in self._module_states[plan[idx]]:
                self._start_gather(st)
        return None

    def _pre_backward(self, mod: nn.Module) -> None:
        states = self._module_states[mod]
        for st in states:
            self._stash_grad(st)
        if self._needs_weight_in_backward(mod):
            for st in states:
                self._finish_gather(st)
            nxt = self._next_in_backward.get(mod)
            if nxt is not None:
                for st in self._module_states[nxt]:
                    self._start_gather(st)
        else:
            for st in states:
                self._materialize_placeholder(st)

    def _free_all(self, states: list[_ShardState]) -> None:
        for st in states:
            self._free(st)

    def _sharded_grad_hook(self, param: nn.Parameter) -> None:
        st = self._states[param]
        full_grad = param.grad
        param.grad = None
        self._free(st)  # the backward of this layer is done: release the full weight
        if full_grad is None:
            return
        if st.reduce_handle is not None:  # same parameter accumulated twice in one backward
            self._finish_reduce(st, keep_as_prev=True)

        reduce_dtype = self.reduce_dtype or st.shard.dtype
        flat = full_grad.reshape(-1).to(reduce_dtype)
        if st.padded_numel != st.numel:
            flat = F.pad(flat, (0, st.padded_numel - st.numel))
        flat = flat.contiguous()
        out = torch.empty(st.shard_numel, dtype=reduce_dtype, device=flat.device)
        st.reduce_in = flat
        st.reduce_out = out
        st.reduce_handle = dist.reduce_scatter_tensor(out, flat, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)

    def _replicated_grad_hook(self, param: nn.Parameter) -> None:
        if param.grad is None:
            return
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)
        self._replicated_handles.append((handle, param.grad))

    def _finish_reduce(self, st: _ShardState, keep_as_prev: bool = False) -> None:
        st.reduce_handle.wait()
        grad = st.reduce_out.div_(self.world_size)
        st.reduce_handle = st.reduce_out = st.reduce_in = None
        if st.prev_grad is not None:
            grad = grad + st.prev_grad if grad.dtype == st.prev_grad.dtype else grad.to(st.prev_grad.dtype) + st.prev_grad
            st.prev_grad = None
        if keep_as_prev:
            st.prev_grad = grad
        else:
            st.param.grad = grad.to(st.param.dtype)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        # Any weights still gathered (e.g. after an interrupted backward) or
        # prefetches that were never consumed are stale by now: drop them.
        for st in self._states.values():
            self._free(st)
        self._exec_order = []

        plan = self._forward_order or self._default_order
        for mod in plan[:2]:
            for st in self._module_states[mod]:
                self._start_gather(st)

        output = self.module(*args, **kwargs)

        if self._exec_order:
            self._forward_order = list(self._exec_order)
            self._build_backward_plan()

        # Prefetch the weights of the first layer of the backward pass.
        if torch.is_grad_enabled() and self._first_in_backward is not None and _requires_grad(output):
            for st in self._module_states[self._first_in_backward]:
                self._start_gather(st)
        return output

    def _build_backward_plan(self) -> None:
        order: list[nn.Module] = []
        for mod in reversed(self._forward_order or []):
            if self._needs_weight_in_backward(mod) and mod not in order:
                order.append(mod)
        self._first_in_backward = order[0] if order else None
        self._next_in_backward = {a: b for a, b in zip(order, order[1:])}

    def finish_gradient_synchronization(self) -> None:
        """Wait for all outstanding gradient communication and install the sharded gradients."""
        for st in self._states.values():
            if st.reduce_handle is not None:
                self._finish_reduce(st)
            elif st.prev_grad is not None:  # parameter received no gradient this time
                st.param.grad = st.prev_grad
                st.prev_grad = None
            self._free(st)  # defensive: nothing should still be gathered here
        for handle, grad in self._replicated_handles:
            handle.wait()
            grad.div_(self.world_size)
        self._replicated_handles.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, Tensor]:
        """All-gather the master shards into full tensors (replicated params are returned as-is)."""
        result: dict[str, Tensor] = {}
        for name, p in self.module.named_parameters():
            st = self._states.get(p)
            if st is None:
                result[name] = p.data
                continue
            shard = st.param.data if not st.is_gathered else st.shard
            full = torch.empty(st.padded_numel, dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(full, shard.contiguous(), group=self.process_group)
            result[name] = full[: st.numel].view(st.shape)
        return result

    def sharded_parameter_names(self) -> list[str]:
        return [st.name for st in self._states.values()]


def _tensors(obj) -> list[Tensor]:
    """Flatten tensors out of a (possibly nested) tuple/list/dict output."""
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in _tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _tensors(o)]
    return []


def _requires_grad(output) -> bool:
    return any(t.requires_grad for t in _tensors(output))
