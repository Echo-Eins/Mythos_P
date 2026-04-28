"""
Dense recurrent-depth causal language model prototype.

This is the first real OpenMythos prototype path.  It intentionally excludes
MoE, MLA, KV-cache, and long-context claims so the core recurrent-depth idea
can be trained and debugged in isolation.  ACT is available as an optional
recurrent-depth halting mode.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.modules import (
    ACTAccumulator,
    ACTHalting,
    AdaRMSNorm,
    AttentionConfig,
    DenseTransformerBlock,
    FFNConfig,
    LTIInjection,
    RMSNorm,
    add_loop_index_embedding,
    init_weights,
)


@dataclass
class DenseLMConfig:
    """
    Configuration for the dense OpenMythos prototype.

    This config is deliberately small and explicit.  The first training runs
    should prove that the model learns and generates before introducing MLA,
    MoE, or KV-cache.  ACT is optional because it changes recurrent-depth
    training dynamics and should be ablated against a fixed-loop baseline.
    """

    vocab_size: int
    dim: int = 1536
    n_heads: int = 24
    n_kv_heads: Optional[int] = None
    prelude_layers: int = 2
    coda_layers: int = 1
    max_loop_iters: int = 4
    max_seq_len: int = 2048
    ffn_hidden_dim: Optional[int] = None
    loop_dim: Optional[int] = None
    dropout: float = 0.0
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-6
    use_ada_norm: bool = True
    lti_init_log_dt: float = -0.5
    lti_init_input_gain: float = 0.3
    lti_init_delta_gain: float = 0.35
    lti_max_input_gain: float = 1.0
    lti_max_delta_gain: float = 1.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    use_act: bool = False
    act_threshold: float = 0.99
    act_ponder_weight: float = 0.01
    act_min_steps: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None

    def resolved_ffn_hidden_dim(self) -> int:
        if self.ffn_hidden_dim is not None:
            return self.ffn_hidden_dim
        # LLaMA-style SwiGLU hidden width.  Rounded to a multiple of 64 for
        # friendlier tensor-core shapes on CUDA.
        raw = int((8 * self.dim) / 3)
        return int(math.ceil(raw / 64) * 64)

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        if self.n_kv_heads is not None and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if self.prelude_layers < 0 or self.coda_layers < 0:
            raise ValueError("prelude_layers and coda_layers must be non-negative")
        if self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.loop_dim is not None and self.loop_dim <= 0:
            raise ValueError("loop_dim must be positive")
        if self.max_loop_iters <= 0:
            raise ValueError("max_loop_iters must be positive")
        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")
        if self.lti_max_input_gain <= 0.0:
            raise ValueError("lti_max_input_gain must be positive")
        if self.lti_max_delta_gain <= 0.0:
            raise ValueError("lti_max_delta_gain must be positive")
        if abs(self.lti_init_input_gain) > self.lti_max_input_gain:
            raise ValueError("abs(lti_init_input_gain) cannot exceed lti_max_input_gain")
        if abs(self.lti_init_delta_gain) > self.lti_max_delta_gain:
            raise ValueError("abs(lti_init_delta_gain) cannot exceed lti_max_delta_gain")
        if not (0.0 < self.act_threshold <= 1.0):
            raise ValueError("act_threshold must be in (0, 1]")
        if self.act_ponder_weight < 0.0:
            raise ValueError("act_ponder_weight must be non-negative")
        if self.act_min_steps <= 0:
            raise ValueError("act_min_steps must be positive")
        if self.act_min_steps > self.max_loop_iters:
            raise ValueError("act_min_steps cannot exceed max_loop_iters")


def dense_lm_config_from_dict(data: dict[str, Any]) -> DenseLMConfig:
    """Reconstruct DenseLMConfig from checkpoint metadata."""

    allowed = DenseLMConfig.__dataclass_fields__.keys()
    values = {k: v for k, v in data.items() if k in allowed}
    if "use_ada_norm" not in values:
        # Old checkpoints were trained with plain RMSNorm.  Preserve their
        # module structure so they can still be loaded for diagnostics.
        values["use_ada_norm"] = False
    return DenseLMConfig(**values)


@dataclass
class DenseLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    stats: Optional[dict[str, Any]] = None


class DenseRecurrentCore(nn.Module):
    """
    Recurrent dense transformer core with LTI input injection.

    Loop equation:

        base_t = norm(h_t + loop_embedding(t))
        delta_t = block(base_t) - base_t
        h_{t+1} = A * h_t + (1 - A) * (B_e * e + B_delta * delta_t)

    If ACT is enabled, the core also predicts a per-token halting probability
    after each recurrent update and returns the ACT-weighted hidden state.  The
    last allowed loop forcibly contributes the remaining probability mass, so
    ACT weights always sum to 1 for every token.
    """

    def __init__(self, cfg: DenseLMConfig) -> None:
        super().__init__()
        ffn_hidden_dim = cfg.resolved_ffn_hidden_dim()
        loop_dim = cfg.loop_dim if cfg.loop_dim is not None else max(2, cfg.dim // 8)
        if loop_dim % 2:
            loop_dim += 1
        if loop_dim > cfg.dim:
            raise ValueError("loop_dim cannot exceed dim")

        self.cfg = cfg
        self.loop_dim = loop_dim
        self.norm = self._make_norm()
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
                hidden_dim=ffn_hidden_dim,
                dropout=cfg.dropout,
            ),
            norm_eps=cfg.norm_eps,
            use_ada_norm=cfg.use_ada_norm,
            ada_cond_dim=loop_dim,
        )
        self.injection = LTIInjection(
            cfg.dim,
            init_log_dt=cfg.lti_init_log_dt,
            init_input_gain=cfg.lti_init_input_gain,
            init_delta_gain=cfg.lti_init_delta_gain,
            max_input_gain=cfg.lti_max_input_gain,
            max_delta_gain=cfg.lti_max_delta_gain,
        )
        self.act_norm = self._make_norm()
        self.act = ACTHalting(cfg.dim)

    def _make_norm(self) -> nn.Module:
        if self.cfg.use_ada_norm:
            return AdaRMSNorm(self.cfg.dim, self.cfg.norm_eps, cond_dim=self.loop_dim)
        return RMSNorm(self.cfg.dim, self.cfg.norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        *,
        n_loops: Optional[int] = None,
        collect_stats: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, Any]]]:
        if h.shape != e.shape:
            raise ValueError("h and e must have identical shapes")
        loops = n_loops or self.cfg.max_loop_iters
        if loops <= 0:
            raise ValueError("n_loops must be positive")

        loop_rms: list[torch.Tensor] = []
        act_p_means: list[torch.Tensor] = []
        accumulator = ACTAccumulator(self.cfg.act_threshold) if self.cfg.use_act else None
        expected_steps: Optional[torch.Tensor] = None
        act_start_loop = min(self.cfg.act_min_steps, loops) - 1

        for loop_index in range(loops):
            h_loop = add_loop_index_embedding(h, loop_index, self.loop_dim)
            norm_cond = h_loop[..., : self.loop_dim]
            base = self.norm(h_loop, norm_cond)
            delta = self.block.forward_delta(
                base,
                cache=None,
                layer_key=f"recurrent.loop_{loop_index}",
                norm_cond=norm_cond,
            )
            h = self.injection(h, e, delta)
            if collect_stats:
                loop_rms.append(h.float().pow(2).mean().sqrt().detach())

            if accumulator is not None and loop_index >= act_start_loop:
                halt_p = self.act(self.act_norm(h, norm_cond))
                act_out, act_weight = accumulator.step(
                    h,
                    halt_p,
                    is_last=(loop_index == loops - 1),
                )
                if expected_steps is None:
                    expected_steps = torch.zeros_like(halt_p)
                expected_steps = expected_steps + act_weight * float(loop_index + 1)
                if collect_stats:
                    act_p_means.append(halt_p.detach().mean())
                if (
                    not self.training
                    and accumulator.halted is not None
                    and bool(accumulator.halted.all())
                ):
                    h = act_out
                    break

        if accumulator is not None:
            assert accumulator.output is not None
            h = accumulator.output

        stats: Optional[dict[str, Any]] = None
        if collect_stats or accumulator is not None:
            stats = {}
            if collect_stats:
                A = self.injection.A().detach()
                input_gain = self.injection.input_gain().detach()
                delta_gain = self.injection.delta_gain().detach()
                retention = 1.0 - A
                stats.update(
                    {
                        "recurrent_loop_rms": torch.stack(loop_rms) if loop_rms else None,
                        "lti_A_min": A.min(),
                        "lti_A_max": A.max(),
                        "lti_tau_min": (-1.0 / A.log()).min(),
                        "lti_tau_max": (-1.0 / A.log()).max(),
                        "lti_B_abs_max": input_gain.abs().max(),
                        "lti_input_gain_abs_max": input_gain.abs().max(),
                        "lti_delta_gain_abs_max": delta_gain.abs().max(),
                        "lti_effective_input_abs_max": (retention * input_gain).abs().max(),
                        "lti_effective_delta_abs_max": (retention * delta_gain).abs().max(),
                    }
                )
            if accumulator is not None:
                assert expected_steps is not None
                assert accumulator.steps is not None
                assert accumulator.halted is not None
                assert accumulator.remainder is not None
                hard_steps = accumulator.steps.float().mean()
                remainder = accumulator.remainder.mean()
                stats.update(
                    {
                        "act_expected_steps": expected_steps.mean(),
                        "act_hard_steps": hard_steps,
                        "act_halt_fraction": accumulator.halted.float().mean().detach(),
                        "act_remainder": remainder,
                        "act_ponder_loss": hard_steps + remainder,
                    }
                )
                if collect_stats and act_p_means:
                    stats["act_halting_p_mean"] = torch.stack(act_p_means).mean()
        return h, stats


class OpenMythosDenseLM(nn.Module):
    """
    Full dense recurrent-depth causal LM.

    This is a complete generator model: it supports training loss, checkpointed
    loading through its config, and autoregressive generation.  Generation
    recomputes the full context at every token by design; KV-cache is excluded
    until the non-cached path is proven correct.
    """

    def __init__(self, cfg: DenseLMConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        ffn_hidden_dim = cfg.resolved_ffn_hidden_dim()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.prelude = nn.ModuleList(
            [
                DenseTransformerBlock(
                    AttentionConfig(
                        dim=cfg.dim,
                        n_heads=cfg.n_heads,
                        n_kv_heads=cfg.n_kv_heads,
                        dropout=cfg.dropout,
                        rope_theta=cfg.rope_theta,
                    ),
                    FFNConfig(
                        dim=cfg.dim,
                        hidden_dim=ffn_hidden_dim,
                        dropout=cfg.dropout,
                    ),
                    norm_eps=cfg.norm_eps,
                    use_ada_norm=cfg.use_ada_norm,
                    ada_cond_dim=None,
                )
                for _ in range(cfg.prelude_layers)
            ]
        )

        self.recurrent = DenseRecurrentCore(cfg)

        self.coda = nn.ModuleList(
            [
                DenseTransformerBlock(
                    AttentionConfig(
                        dim=cfg.dim,
                        n_heads=cfg.n_heads,
                        n_kv_heads=cfg.n_kv_heads,
                        dropout=cfg.dropout,
                        rope_theta=cfg.rope_theta,
                    ),
                    FFNConfig(
                        dim=cfg.dim,
                        hidden_dim=ffn_hidden_dim,
                        dropout=cfg.dropout,
                    ),
                    norm_eps=cfg.norm_eps,
                    use_ada_norm=cfg.use_ada_norm,
                    ada_cond_dim=None,
                )
                for _ in range(cfg.coda_layers)
            ]
        )

        self.norm = (
            AdaRMSNorm(cfg.dim, cfg.norm_eps, cond_dim=None)
            if cfg.use_ada_norm
            else RMSNorm(cfg.dim, cfg.norm_eps)
        )
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(lambda module: init_weights(module, std=cfg.init_std))

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        token_embedding = self.embed.weight.numel()
        lm_head = 0 if self.cfg.tie_embeddings else count(self.lm_head)
        prelude = count(self.prelude)
        recurrent = count(self.recurrent)
        coda = count(self.coda)
        final_norm = count(self.norm)
        body = prelude + recurrent + coda + final_norm
        return {
            "total": self.num_parameters(),
            "body": body,
            "token_embedding": token_embedding,
            "lm_head": lm_head,
            "prelude": prelude,
            "recurrent": recurrent,
            "coda": coda,
            "final_norm": final_norm,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        collect_stats: bool = False,
    ) -> DenseLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, T)")
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        x = self.embed(input_ids)
        for idx, block in enumerate(self.prelude):
            x = block(x, cache=None, layer_key=f"prelude.{idx}")

        e = x
        x, recurrent_stats = self.recurrent(
            x,
            e,
            n_loops=n_loops,
            collect_stats=collect_stats,
        )

        for idx, block in enumerate(self.coda):
            x = block(x, cache=None, layer_key=f"coda.{idx}")

        logits = self.lm_head(self.norm(x))

        loss = None
        stats: Optional[dict[str, Any]] = (
            dict(recurrent_stats) if recurrent_stats is not None else {}
        )
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            lm_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            stats["lm_loss"] = lm_loss.detach()
            if self.cfg.use_act and self.cfg.act_ponder_weight > 0.0:
                if stats is None or "act_ponder_loss" not in stats:
                    raise RuntimeError("ACT is enabled but recurrent core returned no ponder loss")
                act_ponder = stats["act_ponder_loss"]
                act_loss = self.cfg.act_ponder_weight * act_ponder
                loss = loss + act_loss
                stats["act_loss"] = act_loss.detach()

        if stats == {}:
            stats = None
        if stats is not None:
            for key, value in list(stats.items()):
                if torch.is_tensor(value) and key not in {"act_ponder_loss"}:
                    stats[key] = value.detach()
            if "act_ponder_loss" in stats and torch.is_tensor(stats["act_ponder_loss"]):
                stats["act_ponder_loss"] = stats["act_ponder_loss"].detach()

        return DenseLMOutput(logits=logits, loss=loss, stats=stats)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation without KV-cache.

        The model recomputes the full context every step.  This is slower but
        removes cache correctness from the first prototype's risk surface.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        eos = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        out = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            context = out[:, -self.cfg.max_seq_len :]
            logits = self(context, n_loops=n_loops).logits[:, -1, :]
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            if eos is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos),
                    next_token,
                )
            out = torch.cat((out, next_token), dim=1)
            if eos is not None:
                finished = finished | (next_token.squeeze(-1) == eos)
                if bool(finished.all()):
                    break

        return out


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(dim=-1, index=sorted_idx, src=sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)
