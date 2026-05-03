"""
Reference building blocks for OpenMythos prototypes.

This module is intentionally self-contained.  It collects the reusable pieces
that are otherwise scattered across experimental model files:

* RMSNorm
* RoPE with lazy position tables
* dense/GQA causal self-attention
* multi-latent attention with compressed KV cache
* optional autoregressive KV cache
* SwiGLU dense FFN
* sparse reference MoE with an explicit balance loss
* LTI-stable recurrent input injection
* loop-index embedding
* ACT accumulation utilities
* dense transformer and recurrent blocks

The code favors mathematical clarity and predictable behavior over maximum
kernel efficiency.  The first prototype should use the dense path only:
RoPE + dense attention + dense FFN + LTI injection + recurrent block.  KV cache,
MoE, and ACT are included here so they can be reviewed and improved in one
place before being introduced into a training model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AttentionConfig:
    """
    Configuration for dense or grouped-query causal self-attention.

    dim:
        Residual-stream width.
    n_heads:
        Number of query heads.
    n_kv_heads:
        Number of key/value heads.  If omitted, this equals n_heads and the
        layer is standard multi-head attention.  If smaller than n_heads, the
        layer is grouped-query attention and KV heads are repeated across query
        groups.
    dropout:
        Attention probability dropout.  Keep this 0 for inference.
    rope_theta:
        RoPE base.  Larger values rotate high dimensions more slowly and are
        commonly used for longer-context models.
    bias:
        Whether projection layers include bias terms.  Most modern decoder-only
        LMs use bias=False.
    """

    dim: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout: float = 0.0
    rope_theta: float = 500_000.0
    bias: bool = False


@dataclass
class MLAConfig:
    """
    Configuration for Multi-Latent Attention.

    MLA compresses the KV path into a latent vector c_kv plus a small shared
    RoPE key.  The full no-RoPE keys and values are reconstructed from c_kv at
    attention time.  This trades projection compute for lower cache memory.

    qk_rope_head_dim must be even because RoPE rotates adjacent feature pairs.
    qk_nope_head_dim and v_head_dim may be different.  The attention logits use
    qk_nope_head_dim + qk_rope_head_dim; the output projection consumes
    n_heads * v_head_dim.
    """

    dim: int
    n_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    dropout: float = 0.0
    rope_theta: float = 500_000.0
    bias: bool = False


@dataclass
class FFNConfig:
    """
    Configuration for a SwiGLU feed-forward block.

    hidden_dim should usually be close to 8/3 * dim for a LLaMA-style SwiGLU
    FFN, or a smaller value when used as an expert inside MoE.
    """

    dim: int
    hidden_dim: int
    dropout: float = 0.0
    bias: bool = False


@dataclass
class MoEConfig:
    """
    Configuration for the reference sparse MoE layer.

    This implementation is correct and easy to inspect, but it is not a fast
    production MoE kernel.  It loops over experts and dispatches tokens by
    boolean indexing.  Use it to prove routing behavior first; replace it with
    a fused/grouped implementation only after the math is locked down.
    """

    dim: int
    n_routed_experts: int
    n_shared_experts: int
    n_experts_per_token: int
    expert_hidden_dim: int
    router_bias_update_rate: float = 1e-3
    balance_loss_weight: float = 1e-2
    dropout: float = 0.0
    bias: bool = False


@dataclass
class RecurrentConfig:
    """
    Configuration for a dense recurrent-depth block.

    The first prototype should keep use_act=False and use_cache=False while
    validating that language-model loss decreases and loop-depth ablations are
    meaningful.
    """

    dim: int
    n_heads: int
    n_kv_heads: Optional[int]
    ffn_hidden_dim: int
    max_loop_iters: int
    loop_dim: Optional[int] = None
    dropout: float = 0.0
    rope_theta: float = 500_000.0
    act_threshold: float = 0.99
    use_ada_norm: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _even(value: int, name: str) -> None:
    _require(value > 0 and value % 2 == 0, f"{name} must be a positive even integer")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square normalization.

    RMSNorm rescales each residual vector by its root mean square without
    subtracting the mean.  It is common in decoder-only LMs because it is cheap,
    stable, and preserves the residual stream's mean information.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del cond
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        return (x.float() * rms).to(dtype=x.dtype) * weight


class AdaRMSNorm(nn.Module):
    """
    RMSNorm with optional adaptive scale/shift conditioning.

    With cond=None it is an ordinary RMSNorm.  With cond supplied, a zero-
    initialized projection predicts per-channel scale and shift, so the module
    starts identical to RMSNorm and learns adaptive modulation only if useful.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.cond_dim = cond_dim
        self.ada_proj = nn.Linear(cond_dim, 2 * dim, bias=True) if cond_dim is not None else None
        if self.ada_proj is not None:
            self.ada_proj._open_mythos_zero_init = True
            nn.init.zeros_(self.ada_proj.weight)
            nn.init.zeros_(self.ada_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        y = (x.float() * rms).to(dtype=x.dtype) * weight
        if cond is None or self.ada_proj is None:
            return y
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        _require(cond.ndim == 3, "AdaRMSNorm cond must have shape (B, T, C) or (B, C)")
        shift, scale = self.ada_proj(cond.to(dtype=x.dtype)).chunk(2, dim=-1)
        return y * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """
    Lazy RoPE table generator.

    Design choice:
        The original project precomputed RoPE buffers up to max_seq_len during
        model construction.  That is fine for 4k/8k contexts, but bad for
        claimed 1M contexts and unnecessary for short experiments.  This class
        stores only inverse frequencies and materializes cos/sin tables for the
        positions actually used by a forward pass.

    Layout:
        RoPE is applied to adjacent feature pairs: (x0, x1), (x2, x3), ...
        This matches the complex-number interpretation used by the original
        apply_rope implementation.
    """

    def __init__(self, head_dim: int, theta: float = 500_000.0) -> None:
        super().__init__()
        _even(head_dim, "head_dim")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        *,
        start_pos: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos/sin tables shaped (seq_len, head_dim/2).

        start_pos lets incremental decoding request positions
        [start_pos, start_pos + seq_len).
        """

        _require(seq_len >= 0, "seq_len must be non-negative")
        _require(start_pos >= 0, "start_pos must be non-negative")
        device = device or self.inv_freq.device
        dtype = dtype or torch.float32
        positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()
        inv_freq = self.inv_freq.to(device=device)
        angles = torch.outer(positions, inv_freq)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply adjacent-pair RoPE to x.

    Args:
        x:
            Tensor shaped (B, T, H, D).  D must be even.
        cos, sin:
            RoPE tables shaped (T, D/2), usually produced by RotaryEmbedding.

    Returns:
        Tensor with the same shape and dtype as x.

    Mathematical rule for each pair:
        [x_even, x_odd] is multiplied by a 2D rotation matrix:

            even' = even * cos - odd * sin
            odd'  = odd  * cos + even * sin
    """

    _require(x.ndim == 4, "x must have shape (B, T, H, D)")
    B, T, H, D = x.shape
    _even(D, "x.shape[-1]")
    _require(cos.shape == (T, D // 2), "cos must have shape (T, D/2)")
    _require(sin.shape == (T, D // 2), "sin must have shape (T, D/2)")

    x_float = x.float()
    x_pair = x_float.view(B, T, H, D // 2, 2)
    even = x_pair[..., 0]
    odd = x_pair[..., 1]
    cos = cos.view(1, T, 1, D // 2)
    sin = sin.view(1, T, 1, D // 2)
    out_even = even * cos - odd * sin
    out_odd = odd * cos + even * sin
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out.to(dtype=x.dtype)


def add_loop_index_embedding(
    h: torch.Tensor,
    loop_index: int,
    loop_dim: int,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """
    Add a sinusoidal recurrence-depth signal to the leading channels.

    This is not sequence RoPE.  It is a depth-position bias that lets one
    weight-shared recurrent block distinguish early refinement steps from late
    refinement steps.  It is deliberately additive and parameter-free, so it can
    be ablated cleanly.  The layout is [sin..., cos...] to match the legacy
    loop index embedding in main.py.
    """

    _require(h.ndim == 3, "h must have shape (B, T, D)")
    _even(loop_dim, "loop_dim")
    _require(loop_dim <= h.shape[-1], "loop_dim cannot exceed hidden dim")
    _require(loop_index >= 0, "loop_index must be non-negative")

    half_dim = loop_dim // 2
    freqs = 1.0 / (
        theta ** (torch.arange(0, half_dim, device=h.device, dtype=h.dtype) / half_dim)
    )
    angles = loop_index * freqs
    loop_signal = torch.cat((angles.sin(), angles.cos()), dim=-1)
    emb = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb[:loop_dim] = loop_signal
    return h + emb.view(1, 1, -1)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


@dataclass
class KVState:
    """
    Stored keys and values for one attention layer.

    Shapes are (B, S, H_kv, D), matching the projection layout before attention
    transposes heads to (B, H, S, D).
    """

    k: torch.Tensor
    v: torch.Tensor


class KVCache:
    """
    Small explicit autoregressive KV cache.

    The cache is kept outside attention modules so generation code controls
    lifecycle and can clear, detach, or inspect layer entries.  It is optional:
    training and the first prototype should pass cache=None.
    """

    def __init__(self, *, detach: bool = True) -> None:
        self.detach = detach
        self.layers: Dict[str, KVState] = {}

    def clear(self) -> None:
        self.layers.clear()

    def get(self, layer_key: str) -> Optional[KVState]:
        return self.layers.get(layer_key)

    def append(self, layer_key: str, k: torch.Tensor, v: torch.Tensor) -> KVState:
        _require(k.shape == v.shape, "k and v must have identical shapes")
        _require(k.ndim == 4, "k/v must have shape (B, T, H_kv, D)")
        if self.detach:
            k = k.detach()
            v = v.detach()

        prev = self.layers.get(layer_key)
        if prev is not None:
            _require(
                prev.k.shape[0] == k.shape[0]
                and prev.k.shape[2] == k.shape[2]
                and prev.k.shape[3] == k.shape[3],
                "KV cache shape mismatch for layer " + layer_key,
            )
            k = torch.cat((prev.k, k), dim=1)
            v = torch.cat((prev.v, v), dim=1)

        state = KVState(k=k, v=v)
        self.layers[layer_key] = state
        return state


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Expand grouped KV heads to query-head count.

    Input shape:  (B, S, H_kv, D)
    Output shape: (B, S, H_kv * groups, D)
    """

    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=2)


def _causal_additive_mask(
    query_len: int,
    key_len: int,
    *,
    start_pos: int,
    key_start_pos: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build an additive causal mask for cached or uncached attention.

    Query i has absolute position start_pos + i.  Key j has absolute position
    key_start_pos + j.  The query may attend only to keys whose absolute
    position is <= the query position.  Disallowed positions get -inf and
    allowed positions get 0.
    """

    q_pos = torch.arange(start_pos, start_pos + query_len, device=device)
    k_pos = torch.arange(key_start_pos, key_start_pos + key_len, device=device)
    allowed = k_pos.view(1, -1) <= q_pos.view(-1, 1)
    mask = torch.zeros((query_len, key_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask.view(1, 1, query_len, key_len)


# ---------------------------------------------------------------------------
# Attention and FFN
# ---------------------------------------------------------------------------


class DenseCausalSelfAttention(nn.Module):
    """
    Dense causal self-attention with optional GQA and optional KV cache.

    Correctness choices:
        * RoPE is applied before keys are written to cache.
        * Full-sequence training uses PyTorch SDPA with is_causal=True so CUDA
          can select efficient kernels.
        * Cached or offset decoding uses an explicit absolute-position mask,
          avoiding the common bug where a T x T mask is used despite S cached
          keys being present.
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()
        _require(cfg.dim % cfg.n_heads == 0, "dim must be divisible by n_heads")
        n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        _require(n_kv_heads > 0, "n_kv_heads must be positive")
        _require(cfg.n_heads % n_kv_heads == 0, "n_kv_heads must divide n_heads")

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        _even(self.head_dim, "head_dim")
        self.groups = cfg.n_heads // n_kv_heads
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.dim, n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.dim, n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=cfg.bias)
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[KVCache] = None,
        layer_key: str = "attn",
        start_pos: int = 0,
    ) -> torch.Tensor:
        _require(x.ndim == 3, "x must have shape (B, T, D)")
        B, T, D = x.shape
        _require(D == self.dim, "x hidden dim does not match attention dim")

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        cos, sin = self.rope(T, start_pos=start_pos, device=x.device, dtype=x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if cache is not None:
            state = cache.append(layer_key, k, v)
            k = state.k
            v = state.v

        key_len = k.shape[1]
        k = _repeat_kv(k, self.groups)
        v = _repeat_kv(v, self.groups)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, S, D)
        v = v.transpose(1, 2)  # (B, H, S, D)

        can_use_native_causal = cache is None and start_pos == 0 and key_len == T
        if can_use_native_causal:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            mask = _causal_additive_mask(
                T,
                key_len,
                start_pos=start_pos,
                key_start_pos=0 if cache is not None else start_pos,
                device=x.device,
                dtype=q.dtype,
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


@dataclass
class MLAState:
    """
    Cached MLA state for one attention layer.

    c_kv:
        Compressed latent, shape (B, S, kv_lora_rank).
    k_rope:
        RoPE-rotated shared positional key, shape
        (B, S, 1, qk_rope_head_dim).  Keeping one shared head is deliberate:
        expanding it to n_heads in the cache destroys much of MLA's memory
        advantage and is not mathematically required.
    """

    c_kv: torch.Tensor
    k_rope: torch.Tensor


class MLACache:
    """
    Explicit autoregressive cache for Multi-Latent Attention.

    This is separate from KVCache because MLA stores latents rather than full K
    and V tensors.  The cache can be reused across decode steps in the same way:
    prefill with the full prompt, then append one or more new tokens per step.
    """

    def __init__(self, *, detach: bool = True) -> None:
        self.detach = detach
        self.layers: Dict[str, MLAState] = {}

    def clear(self) -> None:
        self.layers.clear()

    def get(self, layer_key: str) -> Optional[MLAState]:
        return self.layers.get(layer_key)

    def append(self, layer_key: str, c_kv: torch.Tensor, k_rope: torch.Tensor) -> MLAState:
        _require(c_kv.ndim == 3, "c_kv must have shape (B, T, R)")
        _require(k_rope.ndim == 4, "k_rope must have shape (B, T, 1, D_rope)")
        _require(c_kv.shape[:2] == k_rope.shape[:2], "MLA cache sequence shape mismatch")
        _require(k_rope.shape[2] == 1, "k_rope must be stored as one shared head")

        if self.detach:
            c_kv = c_kv.detach()
            k_rope = k_rope.detach()

        prev = self.layers.get(layer_key)
        if prev is not None:
            _require(
                prev.c_kv.shape[0] == c_kv.shape[0]
                and prev.c_kv.shape[2] == c_kv.shape[2]
                and prev.k_rope.shape[0] == k_rope.shape[0]
                and prev.k_rope.shape[2:] == k_rope.shape[2:],
                "MLA cache shape mismatch for layer " + layer_key,
            )
            c_kv = torch.cat((prev.c_kv, c_kv), dim=1)
            k_rope = torch.cat((prev.k_rope, k_rope), dim=1)

        state = MLAState(c_kv=c_kv, k_rope=k_rope)
        self.layers[layer_key] = state
        return state


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention, corrected from the original experimental version.

    Original implementation shape:
        x -> q_down/q_norm -> q_nope and q_rope
        x -> kv_down -> c_kv and k_rope_raw
        c_kv -> kv_norm/kv_up -> k_nope and v
        attention(cat(q_nope, rope(q_rope)), cat(k_nope, rope(k_rope_raw)), v)

    Corrections made here:
        * RoPE tables are lazy, not preallocated to max_seq_len.
        * Cached k_rope is stored as a single shared positional head, not
          duplicated across all query heads.
        * Cached attention uses absolute-position masking, so chunked decode
          and prefill do not reuse an invalid T x T mask.
        * The cache type is explicit (`MLACache`) and cannot be confused with
          full K/V cache entries.

    This implementation is still a reference implementation.  For large-scale
    training/inference, profile whether reconstructing K/V from c_kv each decode
    step is cheaper than storing expanded K/V for your exact GB10 setup.
    """

    def __init__(self, cfg: MLAConfig) -> None:
        super().__init__()
        _require(cfg.dim > 0, "dim must be positive")
        _require(cfg.n_heads > 0, "n_heads must be positive")
        _require(cfg.kv_lora_rank > 0, "kv_lora_rank must be positive")
        _require(cfg.q_lora_rank > 0, "q_lora_rank must be positive")
        _require(cfg.qk_nope_head_dim > 0, "qk_nope_head_dim must be positive")
        _require(cfg.v_head_dim > 0, "v_head_dim must be positive")
        _even(cfg.qk_rope_head_dim, "qk_rope_head_dim")

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.q_lora_rank = cfg.q_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.dropout = cfg.dropout

        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=cfg.bias)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank,
            cfg.n_heads * cfg.qk_nope_head_dim,
            bias=cfg.bias,
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank,
            cfg.n_heads * cfg.qk_rope_head_dim,
            bias=cfg.bias,
        )

        self.kv_down = nn.Linear(
            cfg.dim,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            bias=cfg.bias,
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=cfg.bias,
        )
        self.o_proj = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=cfg.bias)
        self.rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.rope_theta)

    def cache_dims_per_token(self) -> int:
        """
        Number of scalar cache entries stored per token per layer.

        This is the honest MLA cache size for this implementation:
        kv_lora_rank + qk_rope_head_dim.  It deliberately does not multiply the
        RoPE component by n_heads because the positional key is shared.
        """

        return self.kv_lora_rank + self.qk_rope_dim

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[MLACache] = None,
        layer_key: str = "mla",
        start_pos: int = 0,
    ) -> torch.Tensor:
        _require(x.ndim == 3, "x must have shape (B, T, D)")
        B, T, D = x.shape
        _require(D == self.dim, "x hidden dim does not match MLA dim")

        cos, sin = self.rope(T, start_pos=start_pos, device=x.device, dtype=x.dtype)

        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        q = torch.cat((q_nope, q_rope), dim=-1)

        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]
        k_rope = kv_raw[..., self.kv_lora_rank :].view(B, T, 1, self.qk_rope_dim)
        k_rope = apply_rotary_emb(k_rope, cos, sin)

        if cache is not None:
            state = cache.append(layer_key, c_kv, k_rope)
            c_kv = state.c_kv
            k_rope = state.k_rope

        key_len = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv))
        kv = kv.view(B, key_len, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]
        v = kv[..., self.qk_nope_dim :]
        k_rope = k_rope.expand(B, key_len, self.n_heads, self.qk_rope_dim)
        k = torch.cat((k_nope, k_rope), dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        can_use_native_causal = cache is None and start_pos == 0 and key_len == T
        if can_use_native_causal:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.qk_head_dim**-0.5,
            )
        else:
            mask = _causal_additive_mask(
                T,
                key_len,
                start_pos=start_pos,
                key_start_pos=0 if cache is not None else start_pos,
                device=x.device,
                dtype=q.dtype,
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.qk_head_dim**-0.5,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_dim)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    Dense SwiGLU feed-forward network.

    SwiGLU computes down(silu(gate(x)) * up(x)).  Compared with a ReLU FFN it is
    smoother and typically gives better language-model quality at similar FLOPs.
    """

    def __init__(self, cfg: FFNConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=cfg.bias)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DenseTransformerBlock(nn.Module):
    """
    Pre-norm transformer block used by the dense prototype.

    forward() returns the full residual output.  forward_delta() returns
    output - input, which is the safer object to feed into an LTI recurrent
    update because the injection already carries A*h + B*e.
    """

    def __init__(
        self,
        attn_cfg: AttentionConfig,
        ffn_cfg: FFNConfig,
        *,
        norm_eps: float = 1e-6,
        use_ada_norm: bool = False,
        ada_cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        _require(attn_cfg.dim == ffn_cfg.dim, "attention and FFN dims must match")
        norm_cls = AdaRMSNorm if use_ada_norm else RMSNorm
        cond_dim = ada_cond_dim if use_ada_norm else None
        self.attn_norm = (
            norm_cls(attn_cfg.dim, norm_eps, cond_dim=cond_dim)
            if use_ada_norm
            else norm_cls(attn_cfg.dim, norm_eps)
        )
        self.ffn_norm = (
            norm_cls(attn_cfg.dim, norm_eps, cond_dim=cond_dim)
            if use_ada_norm
            else norm_cls(attn_cfg.dim, norm_eps)
        )
        self.attn = DenseCausalSelfAttention(attn_cfg)
        self.ffn = SwiGLUFFN(ffn_cfg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[KVCache] = None,
        layer_key: str = "block",
        start_pos: int = 0,
        norm_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x, norm_cond),
            cache=cache,
            layer_key=f"{layer_key}.attn",
            start_pos=start_pos,
        )
        x = x + self.ffn(self.ffn_norm(x, norm_cond))
        return x

    def forward_delta(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[KVCache] = None,
        layer_key: str = "block",
        start_pos: int = 0,
        norm_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            x,
            cache=cache,
            layer_key=layer_key,
            start_pos=start_pos,
            norm_cond=norm_cond,
        ) - x


# ---------------------------------------------------------------------------
# LTI-stable recurrent injection
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable diagonal linear-time-invariant injection with decoupled delta path.

    Recurrent update:

        h_{t+1} = A * h_t
                + (1 - A) * input_gain * e
                + delta_gain * delta_normed

    where delta_normed is delta rescaled to match h's local RMS so the
    transformer residual always operates at h's magnitude, regardless of how
    much the prelude normalization shrinks the transformer's input.

    Decoupling delta from the (1-A) gate is the critical fix: in the old
    convex-combination formula both e and delta were gated by (1-A), so a
    slow-forgetting channel (A near 1, small 1-A) could barely be influenced
    by the transformer at all.  Now (1-A) only applies to the input anchor e
    (which correctly provides weak influence for slow channels) while delta
    contributes a full-strength residual scaled to h's current magnitude.

    Parameters
    ----------
    log_dt: per-channel (shape [dim]) so each channel can learn its own
        time constant independently.  This replaces the old shared scalar.
    log_A: per-channel base memory exponent.  A = exp(-exp(log_dt + log_A)).
    input_gain, delta_gain: tanh-bounded per-channel gains with configurable
        maximum absolute value.  delta_gain may exceed 1.0; the default
        max_delta_gain of 4.0 allows the model to inject transformer updates
        that are 4× the scale of the normalized delta signal.
    """

    def __init__(
        self,
        dim: int,
        *,
        init_log_a: float = 0.0,
        init_log_dt: float = -0.5,
        init_input_gain: float = 0.3,
        init_delta_gain: float = 0.5,
        max_input_gain: float = 1.0,
        max_delta_gain: float = 4.0,
    ) -> None:
        super().__init__()
        _require(max_input_gain > 0.0, "max_input_gain must be positive")
        _require(max_delta_gain > 0.0, "max_delta_gain must be positive")
        _require(
            abs(init_input_gain) <= max_input_gain,
            "abs(init_input_gain) must not exceed max_input_gain",
        )
        _require(
            abs(init_delta_gain) <= max_delta_gain,
            "abs(init_delta_gain) must not exceed max_delta_gain",
        )
        self.log_A = nn.Parameter(torch.full((dim,), init_log_a))
        # Per-channel log_dt for independent time constants across dimensions.
        self.log_dt = nn.Parameter(torch.full((dim,), float(init_log_dt)))
        input_ratio = max(-0.999, min(0.999, init_input_gain / max_input_gain))
        delta_ratio = max(-0.999, min(0.999, init_delta_gain / max_delta_gain))
        self.raw_input_gain = nn.Parameter(torch.full((dim,), math.atanh(input_ratio)))
        self.raw_delta_gain = nn.Parameter(torch.full((dim,), math.atanh(delta_ratio)))
        self.max_input_gain = max_input_gain
        self.max_delta_gain = max_delta_gain

    def A(self) -> torch.Tensor:
        exponent = (self.log_dt + self.log_A).clamp(min=-20.0, max=20.0)
        return torch.exp(-torch.exp(exponent))

    def B(self) -> torch.Tensor:
        return self.input_gain()

    def input_gain(self) -> torch.Tensor:
        return self.max_input_gain * torch.tanh(self.raw_input_gain)

    def delta_gain(self) -> torch.Tensor:
        return self.max_delta_gain * torch.tanh(self.raw_delta_gain)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        _require(h.shape == e.shape == delta.shape, "h, e, and delta shapes must match")
        A = self.A().to(dtype=h.dtype, device=h.device)
        one_minus_A = (1.0 - A).clamp(min=0.0, max=1.0)
        input_gain = self.input_gain().to(dtype=h.dtype, device=h.device)
        delta_gain = self.delta_gain().to(dtype=h.dtype, device=h.device)

        # Rescale delta to h's local RMS so the transformer residual always
        # contributes at h's magnitude.  The scale ratio is detached so it acts
        # as a stable normalization constant rather than an additional gradient
        # path that could amplify unstable delta signals.
        h_rms = h.detach().float().pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        delta_rms = delta.detach().float().pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        delta_normed = (delta.float() * (h_rms / delta_rms)).to(h.dtype)

        # (1-A) gates only the e anchor; delta_normed is a true residual update.
        return (
            A.view(1, 1, -1) * h
            + one_minus_A.view(1, 1, -1) * input_gain.view(1, 1, -1) * e
            + delta_gain.view(1, 1, -1) * delta_normed
        )


# ---------------------------------------------------------------------------
# ACT accumulation
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Per-token Adaptive Computation Time halting probability.

    This module only predicts p_t in (0, 1).  Actual accumulation is handled by
    ACTAccumulator so the remainder logic is explicit and testable.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h)).squeeze(-1)


class ACTAccumulator:
    """
    Stateful ACT weighted-sum accumulator.

    Correctness details:
        * A token contributes p_t while cumulative probability is below the
          threshold.
        * On the halting step it contributes the exact remaining mass.
        * On the final allowed loop, tokens that never reached the threshold are
          forced to contribute their remaining mass.  This prevents returning a
          hidden state whose ACT weights sum to less than 1.

    This does not reduce compute by itself.  It only computes a weighted hidden
    state.  Real compute savings require a separate scheduler or masking kernel.
    """

    def __init__(self, threshold: float) -> None:
        _require(0.0 < threshold <= 1.0, "ACT threshold must be in (0, 1]")
        self.threshold = threshold
        self.cumulative: Optional[torch.Tensor] = None
        self.halted: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.steps: Optional[torch.Tensor] = None
        self.remainder: Optional[torch.Tensor] = None

    def step(
        self,
        h: torch.Tensor,
        p: torch.Tensor,
        *,
        is_last: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _require(h.ndim == 3, "h must have shape (B, T, D)")
        _require(p.shape == h.shape[:2], "p must have shape (B, T)")

        if self.cumulative is None:
            self.cumulative = torch.zeros_like(p)
            self.halted = torch.zeros_like(p, dtype=torch.bool)
            self.output = torch.zeros_like(h)
            self.steps = torch.zeros_like(p)
            self.remainder = torch.zeros_like(p)

        assert self.halted is not None
        assert self.output is not None
        assert self.steps is not None
        assert self.cumulative is not None
        assert self.remainder is not None

        running = ~self.halted
        remaining = (1.0 - self.cumulative).clamp(min=0.0, max=1.0)
        crosses = self.cumulative + p >= self.threshold
        halting_now = running & (crosses | is_last)

        weight = torch.where(crosses | is_last, remaining, p)
        weight = weight * running.to(dtype=weight.dtype)

        self.output = self.output + weight.unsqueeze(-1) * h
        self.cumulative = self.cumulative + weight
        self.steps = self.steps + running.to(dtype=self.steps.dtype)
        self.remainder = torch.where(halting_now, remaining, self.remainder)
        self.halted = self.halted | crosses | is_last
        return self.output, weight


# ---------------------------------------------------------------------------
# Reference MoE
# ---------------------------------------------------------------------------


@dataclass
class MoEOutput:
    x: torch.Tensor
    balance_loss: Optional[torch.Tensor]
    expert_load: torch.Tensor


class MoERouter(nn.Module):
    """
    Top-k router with unbiased gate weights and optional routing bias.

    Selection can use router_bias to encourage underused experts.  The selected
    expert weights are gathered from the unbiased softmax probabilities, then
    renormalized over the selected experts.  This mirrors the aux-loss-free
    bias idea while keeping gradient semantics clean.
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        _require(cfg.n_experts_per_token > 0, "top-k must be positive")
        _require(
            cfg.n_experts_per_token <= cfg.n_routed_experts,
            "top-k cannot exceed routed expert count",
        )
        self.n_routed_experts = cfg.n_routed_experts
        self.top_k = cfg.n_experts_per_token
        self.balance_loss_weight = cfg.balance_loss_weight
        self.bias_update_rate = cfg.router_bias_update_rate
        self.linear = nn.Linear(cfg.dim, cfg.n_routed_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_routed_experts))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(x.float())
        probs = logits.softmax(dim=-1)
        _, indices = (logits + self.router_bias).topk(self.top_k, dim=-1)
        weights = probs.gather(dim=-1, index=indices)
        if self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        one_hot = F.one_hot(indices, num_classes=self.n_routed_experts).float()
        expert_load = one_hot.sum(dim=(0, 1)) if one_hot.ndim == 3 else one_hot.sum(0)
        return indices, weights.to(dtype=x.dtype), probs, expert_load

    def balance_loss(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Switch/DeepSeek-style expert balance loss.

        expert_fraction is the fraction of top-k assignments going to each
        expert.  probability_fraction is the mean soft router probability.
        The product is minimized near uniform routing and grows when high-prob
        experts also receive too many hard assignments.
        """

        flat_indices = indices.reshape(-1)
        counts = F.one_hot(flat_indices, num_classes=self.n_routed_experts).float()
        expert_fraction = counts.mean(dim=0)
        probability_fraction = probs.reshape(-1, self.n_routed_experts).mean(dim=0)
        loss = self.n_routed_experts * (expert_fraction * probability_fraction).sum()
        return self.balance_loss_weight * loss

    @torch.no_grad()
    def update_router_bias(self, expert_load: torch.Tensor) -> None:
        """
        Optional aux-loss-free bias update.

        Underused experts receive a positive bias; overused experts receive a
        negative bias.  This should be called from the training loop after a
        step's loads are reduced across data-parallel ranks.
        """

        load = expert_load.float()
        if load.sum() <= 0:
            return
        target = load.mean()
        direction = torch.sign(target - load)
        self.router_bias.add_(self.bias_update_rate * direction)


class SparseMoE(nn.Module):
    """
    Reference sparse MoE layer with shared and routed SwiGLU experts.

    This is included for correctness review, not for the first dense prototype.
    It returns the balance loss and hard expert load so training code can log
    routing collapse immediately.
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = MoERouter(cfg)
        self.routed = nn.ModuleList(
            [
                SwiGLUFFN(
                    FFNConfig(
                        dim=cfg.dim,
                        hidden_dim=cfg.expert_hidden_dim,
                        dropout=cfg.dropout,
                        bias=cfg.bias,
                    )
                )
                for _ in range(cfg.n_routed_experts)
            ]
        )
        shared_hidden = cfg.n_shared_experts * cfg.expert_hidden_dim
        self.shared = (
            SwiGLUFFN(
                FFNConfig(
                    dim=cfg.dim,
                    hidden_dim=shared_hidden,
                    dropout=cfg.dropout,
                    bias=cfg.bias,
                )
            )
            if cfg.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> MoEOutput:
        shape = x.shape
        _require(x.ndim == 3, "x must have shape (B, T, D)")
        flat = x.reshape(-1, shape[-1])

        indices, weights, probs, expert_load = self.router(flat)
        routed_out = torch.zeros_like(flat)
        flat_indices = indices.reshape(-1)

        for expert_id, expert in enumerate(self.routed):
            token_slot = torch.where(flat_indices == expert_id)[0]
            if token_slot.numel() == 0:
                continue
            token_idx = token_slot // self.cfg.n_experts_per_token
            rank_idx = token_slot % self.cfg.n_experts_per_token
            expert_out = expert(flat[token_idx]) * weights[token_idx, rank_idx].unsqueeze(-1)
            routed_out.index_add_(0, token_idx, expert_out)

        if self.shared is not None:
            routed_out = routed_out + self.shared(flat)

        balance = (
            self.router.balance_loss(probs, indices)
            if self.training and self.cfg.balance_loss_weight > 0.0
            else None
        )
        return MoEOutput(
            x=routed_out.view(shape),
            balance_loss=balance,
            expert_load=expert_load,
        )


# ---------------------------------------------------------------------------
# Dense recurrent block
# ---------------------------------------------------------------------------


class DenseRecurrentBlock(nn.Module):
    """
    Dense recurrent-depth block for the first prototype.

    Loop body:

        base_t  = norm(h_t + loop_embedding(t))
        delta_t = TransformerBlock(base_t) - base_t
        h_{t+1} = A * h_t + (1 - A) * (B_e * e + B_delta * delta_t)

    The key decision is using the transformer's residual delta rather than its
    full output.  LTI injection already carries the previous hidden state and
    the encoded input; adding the block's full residual output would duplicate
    that base signal and make loop stability harder to reason about.
    """

    def __init__(self, cfg: RecurrentConfig, *, norm_eps: float = 1e-6) -> None:
        super().__init__()
        loop_dim = cfg.loop_dim if cfg.loop_dim is not None else max(2, cfg.dim // 8)
        if loop_dim % 2:
            loop_dim += 1
        _require(loop_dim <= cfg.dim, "loop_dim cannot exceed dim")

        self.cfg = cfg
        self.loop_dim = loop_dim
        self.norm = (
            AdaRMSNorm(cfg.dim, norm_eps, cond_dim=loop_dim)
            if cfg.use_ada_norm
            else RMSNorm(cfg.dim, norm_eps)
        )
        self.block = DenseTransformerBlock(
            AttentionConfig(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout=cfg.dropout,
                rope_theta=cfg.rope_theta,
            ),
            FFNConfig(
                dim=cfg.dim,
                hidden_dim=cfg.ffn_hidden_dim,
                dropout=cfg.dropout,
            ),
            norm_eps=norm_eps,
            use_ada_norm=cfg.use_ada_norm,
            ada_cond_dim=loop_dim,
        )
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        *,
        n_loops: Optional[int] = None,
        use_act: bool = False,
        cache: Optional[KVCache] = None,
        layer_key: str = "recurrent",
        start_pos: int = 0,
    ) -> torch.Tensor:
        _require(h.shape == e.shape, "h and e shapes must match")
        loops = n_loops or self.cfg.max_loop_iters
        _require(loops > 0, "n_loops must be positive")

        accumulator = ACTAccumulator(self.cfg.act_threshold) if use_act else None

        for loop_index in range(loops):
            h_loop = add_loop_index_embedding(h, loop_index, self.loop_dim)
            norm_cond = h_loop[..., : self.loop_dim]
            base = self.norm(h_loop, norm_cond)
            delta = self.block.forward_delta(
                base,
                cache=cache,
                layer_key=f"{layer_key}.loop_{loop_index}",
                start_pos=start_pos,
                norm_cond=norm_cond,
            )
            h = self.injection(h, e, delta)

            if accumulator is not None:
                p = self.act(h)
                out, _ = accumulator.step(h, p, is_last=(loop_index == loops - 1))
                if (
                    cache is None
                    and accumulator.halted is not None
                    and bool(accumulator.halted.all())
                ):
                    return out

        if accumulator is not None:
            assert accumulator.output is not None
            return accumulator.output
        return h


def init_weights(module: nn.Module, *, std: float = 0.02) -> None:
    """
    GPT-style initialization helper.

    Keep this explicit instead of hiding initialization inside every block.  It
    lets prototype models choose their own scaling policy later.
    """

    if isinstance(module, nn.Linear):
        if getattr(module, "_open_mythos_zero_init", False):
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
