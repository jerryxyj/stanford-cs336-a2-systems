"""FlashAttention-2 (Dao, 2023) implementations.

Two ``torch.autograd.Function`` subclasses are provided:

* :class:`FlashAttentionPyTorch` -- a tiled, pure-PyTorch implementation of the
  FlashAttention-2 forward pass (Algorithm 1 of the handout). It is slow, but
  easy to debug and runs on CPU.
* :class:`FlashAttentionTriton` -- the same algorithm as a fused Triton kernel.
  By default the backward pass is the recomputation-based PyTorch backward
  (Equations 13-19 of the handout) wrapped in ``torch.compile``. Setting
  ``FlashAttentionTriton.USE_TRITON_BACKWARD = True`` switches to the tiled
  Triton backward of Algorithm 2 (two passes: one for dK/dV, one for dQ, which
  avoids atomics).

Both functions take ``(Q, K, V, is_causal=False)`` where ``Q`` has shape
``(..., n_queries, d)`` and ``K``/``V`` have shape ``(..., n_keys, d)``. All
leading dimensions are flattened into a single batch dimension. The forward pass
saves ``L`` (the row-wise logsumexp of the attention scores), ``Q``, ``K``,
``V`` and ``O`` for the backward pass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

try:  # Triton is only available on Linux + CUDA.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on machines without a GPU
    triton = None
    tl = None
    HAS_TRITON = False


MASK_VALUE = -1e6  # Added to masked-out attention scores (finite to avoid NaNs).


def _flatten_batch(x: Tensor) -> Tensor:
    """(..., n, d) -> (batch, n, d), with batch = prod(...)."""
    return x.reshape(-1, x.shape[-2], x.shape[-1])


# --------------------------------------------------------------------------- #
# Pure PyTorch forward (Algorithm 1)                                           #
# --------------------------------------------------------------------------- #
def flash_attention_forward_pytorch(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    is_causal: bool = False,
    q_tile_size: int = 64,
    k_tile_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """Tiled FlashAttention-2 forward pass.

    Args:
        Q: (batch, n_queries, d)
        K, V: (batch, n_keys, d)
    Returns:
        O: (batch, n_queries, d) attention output, same dtype as Q
        L: (batch, n_queries) float32 logsumexp of the (scaled, masked) scores
    """
    batch, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = 1.0 / math.sqrt(d)
    bq = min(q_tile_size, n_queries)
    bk = min(k_tile_size, n_keys)

    O = torch.empty_like(Q)
    L = torch.empty(batch, n_queries, device=Q.device, dtype=torch.float32)

    for q_start in range(0, n_queries, bq):
        q_end = q_start + bq
        Qi = Q[:, q_start:q_end]
        q_idx = torch.arange(q_start, q_end, device=Q.device)

        Oi = torch.zeros(batch, bq, d, device=Q.device, dtype=torch.float32)
        li = torch.zeros(batch, bq, device=Q.device, dtype=torch.float32)
        mi = torch.full((batch, bq), float("-inf"), device=Q.device, dtype=torch.float32)

        # With causal masking, key tiles that lie entirely above the diagonal
        # contribute exactly zero and can be skipped.
        n_keys_needed = min(q_end, n_keys) if is_causal else n_keys
        for k_start in range(0, n_keys_needed, bk):
            k_end = k_start + bk
            Kj = K[:, k_start:k_end]
            Vj = V[:, k_start:k_end]

            Sij = torch.einsum("bqd,bkd->bqk", Qi, Kj).float() * scale
            if is_causal:
                k_idx = torch.arange(k_start, k_end, device=Q.device)
                Sij = torch.where(q_idx[:, None] >= k_idx[None, :], Sij, Sij + MASK_VALUE)

            m_new = torch.maximum(mi, Sij.amax(dim=-1))
            P_tilde = torch.exp(Sij - m_new[..., None])
            alpha = torch.exp(mi - m_new)
            li = alpha * li + P_tilde.sum(dim=-1)
            Oi = alpha[..., None] * Oi + torch.einsum("bqk,bkd->bqd", P_tilde.to(Vj.dtype), Vj).float()
            mi = m_new

        O[:, q_start:q_end] = (Oi / li[..., None]).to(O.dtype)
        L[:, q_start:q_end] = mi + torch.log(li)

    return O, L


# --------------------------------------------------------------------------- #
# Backward with recomputation (Equations 13-19), optionally torch.compile'd    #
# --------------------------------------------------------------------------- #
def flash_attention_backward_torch(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    O: Tensor,
    dO: Tensor,
    L: Tensor,
    is_causal: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute P from (Q, K, L) and compute dQ, dK, dV without a softmax backward."""
    n_queries, n_keys = Q.shape[-2], K.shape[-2]
    # D = rowsum(O * dO) == rowsum(P * dP)
    D = (O.float() * dO.float()).sum(dim=-1)  # (batch, n_queries)

    S = torch.einsum("bqd,bkd->bqk", Q, K).float() * scale
    if is_causal:
        q_idx = torch.arange(n_queries, device=Q.device)
        k_idx = torch.arange(n_keys, device=Q.device)
        S = torch.where(q_idx[:, None] >= k_idx[None, :], S, S + MASK_VALUE)
    P = torch.exp(S - L[..., None])  # (batch, n_queries, n_keys), float32

    dV = torch.einsum("bqk,bqd->bkd", P.to(dO.dtype), dO)
    dP = torch.einsum("bqd,bkd->bqk", dO, V).float()
    dS = P * (dP - D[..., None])
    dQ = torch.einsum("bqk,bkd->bqd", dS.to(K.dtype), K) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS.to(Q.dtype), Q) * scale
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


# torch.compile is used on CUDA by default (where it fuses the elementwise ops
# into the matmuls). On CPU it works too, but compilation takes a while, so it
# is opt-in there.
COMPILE_BACKWARD_ON_CPU = False
_compiled_backward = None


def _get_torch_backward(device: torch.device):
    global _compiled_backward
    if device.type == "cuda" or COMPILE_BACKWARD_ON_CPU:
        if _compiled_backward is None:
            _compiled_backward = torch.compile(flash_attention_backward_torch)
        return _compiled_backward
    return flash_attention_backward_torch


class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 with a tiled pure-PyTorch forward and recomputation backward."""

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        lead_shape = Q.shape[:-2]
        Qf, Kf, Vf = _flatten_batch(Q), _flatten_batch(K), _flatten_batch(V)
        O, L = flash_attention_forward_pytorch(Qf, Kf, Vf, is_causal, FlashAttentionPyTorch.Q_TILE_SIZE, FlashAttentionPyTorch.K_TILE_SIZE)
        O = O.reshape(Q.shape)
        L = L.reshape(*lead_shape, Q.shape[-2])
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])
        bwd = _get_torch_backward(Q.device)
        dQ, dK, dV = bwd(
            _flatten_batch(Q),
            _flatten_batch(K),
            _flatten_batch(V),
            _flatten_batch(O),
            _flatten_batch(dO.contiguous()),
            L.reshape(-1, L.shape[-1]),
            ctx.is_causal,
            scale,
        )
        return dQ.reshape(Q.shape), dK.reshape(K.shape), dV.reshape(V.shape), None


# --------------------------------------------------------------------------- #
# Triton kernels                                                               #
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        # Program indices: one program per (query tile, batch element).
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor.
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        q = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        # Running statistics / output accumulator, all kept in float32.
        m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        if is_causal:
            # Key tiles strictly above the diagonal are fully masked: skip them.
            n_key_tiles = tl.cdiv(tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS), K_TILE_SIZE)
        else:
            n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(0, n_key_tiles):
            k = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
            v = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

            s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE), float32
            if is_causal:
                k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                s = tl.where(q_offsets[:, None] >= k_offsets[None, :], s, s - 1.0e6)  # add MASK_VALUE

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        acc = acc / l_i[:, None]
        l_final = m_i + tl.log(l_i)
        tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty))
        tl.store(L_block_ptr, l_final.to(L_block_ptr.type.element_ty))

    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        D_ptr,
        dK_ptr,
        dV_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_db,
        stride_dq,
        stride_dkb,
        stride_dkk,
        stride_dkd,
        stride_dvb,
        stride_dvk,
        stride_dvd,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """First backward pass (Algorithm 2, lines 6-19): one program per key tile computes dK, dV."""
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        if is_causal:
            # Query tiles that end before this key tile starts are fully masked.
            q_tile_start = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
        else:
            q_tile_start = 0

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(q_tile_start * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(q_tile_start * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(q_tile_start * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(q_tile_start * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        v = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        dk = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

        n_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
        for i in range(q_tile_start, n_query_tiles):
            q = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)
            do = tl.load(dO_block_ptr)  # (Q_TILE_SIZE, D)
            l_i = tl.load(L_block_ptr)  # (Q_TILE_SIZE,)
            d_i = tl.load(D_block_ptr)  # (Q_TILE_SIZE,)

            s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
            if is_causal:
                q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                s = tl.where(q_offsets[:, None] >= k_offsets[None, :], s, s - 1.0e6)  # add MASK_VALUE
            p = tl.exp(s - l_i[:, None])  # recomputed attention probabilities

            dv = tl.dot(tl.trans(p).to(do.dtype), do, dv)  # (K_TILE_SIZE, D)
            dp = tl.dot(do, tl.trans(v))  # (Q_TILE_SIZE, K_TILE_SIZE)
            ds = p * (dp - d_i[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk)  # (K_TILE_SIZE, D), scaled below

            Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
            dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
            L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
            D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

        dk = dk * scale
        tl.store(dK_block_ptr, dk.to(dK_block_ptr.type.element_ty))
        tl.store(dV_block_ptr, dv.to(dV_block_ptr.type.element_ty))

    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        L_ptr,
        D_ptr,
        dQ_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_lb,
        stride_lq,
        stride_db,
        stride_dq,
        stride_dqb,
        stride_dqq,
        stride_dqd,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """Second backward pass (Algorithm 2, lines 20-32): one program per query tile computes dQ."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)
        do = tl.load(dO_block_ptr)
        l_i = tl.load(L_block_ptr)
        d_i = tl.load(D_block_ptr)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        if is_causal:
            n_key_tiles = tl.cdiv(tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS), K_TILE_SIZE)
        else:
            n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(0, n_key_tiles):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            s = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                s = tl.where(q_offsets[:, None] >= k_offsets[None, :], s, s - 1.0e6)  # add MASK_VALUE
            p = tl.exp(s - l_i[:, None])

            dp = tl.dot(do, tl.trans(v))  # (Q_TILE_SIZE, K_TILE_SIZE)
            ds = p * (dp - d_i[:, None])
            dq = tl.dot(ds.to(k.dtype), k, dq)  # (Q_TILE_SIZE, D), scaled below

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        dq = dq * scale
        tl.store(dQ_block_ptr, dq.to(dQ_block_ptr.type.element_ty))


def _pick_tile(requested: int, n: int) -> int:
    """Largest tile <= requested that divides n (n is a power of two >= 16 in our tests)."""
    tile = min(requested, n)
    while n % tile != 0:
        tile //= 2
    assert tile >= 16, f"tile size must be >= 16, got {tile} for sequence length {n}"
    return tile


def flash_attention_forward_triton(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool, q_tile_size: int, k_tile_size: int) -> tuple[Tensor, Tensor]:
    """Launch the fused forward kernel. Inputs are (batch, n, d) CUDA tensors."""
    assert HAS_TRITON, "Triton is not available"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernels require CUDA tensors"
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    batch, n_queries, d = Q.shape
    n_keys = K.shape[1]
    assert d >= 16 and (d & (d - 1)) == 0, "head dimension must be a power of two >= 16"

    bq = _pick_tile(q_tile_size, n_queries)
    bk = _pick_tile(k_tile_size, n_keys)

    O = torch.empty_like(Q)
    L = torch.empty(batch, n_queries, device=Q.device, dtype=torch.float32)

    grid = (triton.cdiv(n_queries, bq), batch)
    flash_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        L,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        L.stride(0),
        L.stride(1),
        n_queries,
        n_keys,
        1.0 / math.sqrt(d),
        D=d,
        Q_TILE_SIZE=bq,
        K_TILE_SIZE=bk,
        is_causal=is_causal,
    )
    return O, L


def flash_attention_backward_triton(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    O: Tensor,
    dO: Tensor,
    L: Tensor,
    is_causal: bool,
    q_tile_size: int,
    k_tile_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch the two tiled backward kernels (Algorithm 2). Inputs are (batch, n, d) CUDA tensors."""
    assert HAS_TRITON, "Triton is not available"
    Q, K, V, O, dO, L = (t.contiguous() for t in (Q, K, V, O, dO, L))
    batch, n_queries, d = Q.shape
    n_keys = K.shape[1]
    scale = 1.0 / math.sqrt(d)

    bq = _pick_tile(q_tile_size, n_queries)
    bk = _pick_tile(k_tile_size, n_keys)

    # D = rowsum(O * dO), computed once in global memory.
    D = (O.float() * dO.float()).sum(dim=-1).contiguous()  # (batch, n_queries)

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    common_strides = (
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        L.stride(0),
        L.stride(1),
        D.stride(0),
        D.stride(1),
    )

    flash_bwd_dkdv_kernel[(triton.cdiv(n_keys, bk), batch)](
        Q,
        K,
        V,
        dO,
        L,
        D,
        dK,
        dV,
        *common_strides,
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dV.stride(0),
        dV.stride(1),
        dV.stride(2),
        n_queries,
        n_keys,
        scale,
        D=d,
        Q_TILE_SIZE=bq,
        K_TILE_SIZE=bk,
        is_causal=is_causal,
    )
    flash_bwd_dq_kernel[(triton.cdiv(n_queries, bq), batch)](
        Q,
        K,
        V,
        dO,
        L,
        D,
        dQ,
        *common_strides,
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
        n_queries,
        n_keys,
        scale,
        D=d,
        Q_TILE_SIZE=bq,
        K_TILE_SIZE=bk,
        is_causal=is_causal,
    )
    return dQ, dK, dV


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a fused Triton forward kernel.

    The backward pass uses the recomputation-based PyTorch implementation under
    ``torch.compile`` by default. Set ``USE_TRITON_BACKWARD = True`` to use the
    tiled Triton backward kernels instead (Section 4.2.3 of the handout).
    """

    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64
    BWD_Q_TILE_SIZE = 64
    BWD_K_TILE_SIZE = 64
    USE_TRITON_BACKWARD = False

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        lead_shape = Q.shape[:-2]
        O, L = flash_attention_forward_triton(
            _flatten_batch(Q),
            _flatten_batch(K),
            _flatten_batch(V),
            is_causal,
            FlashAttentionTriton.Q_TILE_SIZE,
            FlashAttentionTriton.K_TILE_SIZE,
        )
        O = O.reshape(Q.shape)
        L = L.reshape(*lead_shape, Q.shape[-2])
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        args = (
            _flatten_batch(Q),
            _flatten_batch(K),
            _flatten_batch(V),
            _flatten_batch(O),
            _flatten_batch(dO.contiguous()),
            L.reshape(-1, L.shape[-1]),
            ctx.is_causal,
        )
        if FlashAttentionTriton.USE_TRITON_BACKWARD:
            dQ, dK, dV = flash_attention_backward_triton(*args, FlashAttentionTriton.BWD_Q_TILE_SIZE, FlashAttentionTriton.BWD_K_TILE_SIZE)
        else:
            dQ, dK, dV = _get_torch_backward(Q.device)(*args, 1.0 / math.sqrt(Q.shape[-1]))
        return dQ.reshape(Q.shape), dK.reshape(K.shape), dV.reshape(V.shape), None


def flash_attention(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False, backend: str = "auto") -> Tensor:
    """Functional entry point. ``backend`` is one of ``"auto"``, ``"triton"``, ``"pytorch"``."""
    if backend == "auto":
        backend = "triton" if (HAS_TRITON and Q.is_cuda) else "pytorch"
    if backend == "triton":
        return FlashAttentionTriton.apply(Q, K, V, is_causal)
    return FlashAttentionPyTorch.apply(Q, K, V, is_causal)
