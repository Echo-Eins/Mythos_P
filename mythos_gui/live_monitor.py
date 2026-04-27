"""Live metric loading and Plotly dashboards for Mythos training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_event(obj: dict[str, Any]) -> dict[str, Any]:
    if "event" in obj:
        return obj
    if "eval" in obj:
        return {"event": "eval", "step": obj.get("step"), "eval": obj["eval"]}
    if "exact_eval" in obj:
        return {"event": "exact_eval", "exact_eval": obj["exact_eval"]}
    if "sample" in obj:
        return {"event": "exact_sample", **obj["sample"]}
    if "loss" in obj and "step" in obj:
        return {"event": "train", **obj}
    if "params" in obj and "train_chunks" in obj:
        return {"event": "startup", **obj}
    return {"event": "raw", **obj}


def load_events(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not str(path).strip():
        return []
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return []

    events: list[dict[str, Any]] = []
    for raw_line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(normalize_event(obj))
    return events


def flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    row = dict(event)
    kind = row.get("event")
    if kind == "eval" and isinstance(row.get("eval"), dict):
        metrics = row.pop("eval")
        for key, value in metrics.items():
            row[f"eval_{key}"] = value
    if kind == "exact_eval" and isinstance(row.get("exact_eval"), dict):
        metrics = row.pop("exact_eval")
        for key, value in metrics.items():
            row[f"exact_{key}"] = value
    if isinstance(row.get("loop_rms"), list):
        for idx, value in enumerate(row["loop_rms"]):
            row[f"loop_rms_{idx + 1}"] = value
    return row


def events_to_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([flatten_event(event) for event in events])


def load_frame(path: str | Path | None) -> pd.DataFrame:
    return events_to_frame(load_events(path))


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        margin=dict(l=48, r=24, t=48, b=40),
    )
    fig.add_annotation(
        text="No data",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    return fig


def _numeric(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().any():
        return series
    return None


def _add_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    name: str,
    row: int,
    col: int,
    mode: str = "lines",
) -> None:
    x = _numeric(df, x_col)
    y = _numeric(df, y_col)
    if x is None or y is None:
        return
    mask = x.notna() & y.notna()
    if not mask.any():
        return
    fig.add_trace(
        go.Scatter(x=x[mask], y=y[mask], name=name, mode=mode),
        row=row,
        col=col,
    )


def training_dashboard(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Training metrics")
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Loss",
            "Perplexity",
            "Learning rate",
            "Throughput",
            "Gradient and memory",
            "ACT",
        ),
    )
    _add_trace(fig, df, x_col="step", y_col="loss", name="train loss", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="lm_loss", name="train LM loss", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="eval_loss", name="eval loss", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="eval_lm_loss", name="eval LM loss", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="eval_ppl", name="eval ppl", row=1, col=2)
    _add_trace(fig, df, x_col="step", y_col="lr", name="lr", row=2, col=1)
    _add_trace(fig, df, x_col="step", y_col="tok_s", name="tokens/s", row=2, col=2)
    _add_trace(fig, df, x_col="step", y_col="epochs_seen", name="epochs seen", row=2, col=2)
    _add_trace(fig, df, x_col="step", y_col="grad_norm", name="grad norm", row=3, col=1)
    _add_trace(fig, df, x_col="step", y_col="cuda_mem_gb", name="CUDA GB", row=3, col=1)
    _add_trace(fig, df, x_col="step", y_col="act_expected_steps", name="expected", row=3, col=2)
    _add_trace(fig, df, x_col="step", y_col="act_hard_steps", name="hard", row=3, col=2)
    _add_trace(fig, df, x_col="step", y_col="act_halt_fraction", name="halted", row=3, col=2)
    fig.update_layout(
        title="Training metrics",
        template="plotly_white",
        height=780,
        margin=dict(l=48, r=24, t=72, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    return fig


def recurrent_dashboard(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Recurrent/LTI diagnostics")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("LTI A", "LTI gains", "Effective LTI gains", "Loop RMS"),
    )
    _add_trace(fig, df, x_col="step", y_col="lti_A_min", name="A min", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="lti_A_max", name="A max", row=1, col=1)
    _add_trace(fig, df, x_col="step", y_col="lti_input_gain_abs_max", name="input", row=1, col=2)
    _add_trace(fig, df, x_col="step", y_col="lti_delta_gain_abs_max", name="delta", row=1, col=2)
    _add_trace(fig, df, x_col="step", y_col="lti_effective_input_abs_max", name="input eff", row=2, col=1)
    _add_trace(fig, df, x_col="step", y_col="lti_effective_delta_abs_max", name="delta eff", row=2, col=1)
    for column in sorted(col for col in df.columns if str(col).startswith("loop_rms_")):
        _add_trace(fig, df, x_col="step", y_col=column, name=column, row=2, col=2)
    fig.update_layout(
        title="Recurrent/LTI diagnostics",
        template="plotly_white",
        height=620,
        margin=dict(l=48, r=24, t=72, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    return fig


def generation_dashboard(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Generation diagnostics")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Exact match", "EOS/max-token behavior", "Length", "Repetition"),
    )
    sample_df = df[df.get("event") == "exact_sample"] if "event" in df.columns else pd.DataFrame()
    summary_df = df[df.get("event") == "exact_eval"] if "event" in df.columns else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df.reset_index().rename(columns={"index": "summary_idx"})
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_exact_match", name="exact", row=1, col=1, mode="markers+lines")
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_eos_rate", name="EOS", row=1, col=2, mode="markers+lines")
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_hit_max_new_tokens_rate", name="hit max", row=1, col=2, mode="markers+lines")
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_avg_generated_tokens", name="avg tokens", row=2, col=1, mode="markers+lines")
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_avg_repeat_4gram_ratio", name="repeat 4g", row=2, col=2, mode="markers+lines")
        _add_trace(fig, summary_df, x_col="summary_idx", y_col="exact_avg_unique_token_ratio", name="unique", row=2, col=2, mode="markers+lines")
    if not sample_df.empty:
        sample_df = sample_df.reset_index().rename(columns={"index": "sample_idx"})
        _add_trace(fig, sample_df, x_col="sample_idx", y_col="generated_tokens", name="sample tokens", row=2, col=1, mode="markers")
        _add_trace(fig, sample_df, x_col="sample_idx", y_col="repeat_4gram_ratio", name="sample repeat", row=2, col=2, mode="markers")
    fig.update_layout(
        title="Generation diagnostics",
        template="plotly_white",
        height=620,
        margin=dict(l=48, r=24, t=72, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    return fig


def summary_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No metric events loaded."
    lines: list[str] = []
    startup = df[df["event"] == "startup"] if "event" in df.columns else pd.DataFrame()
    if not startup.empty:
        latest = startup.iloc[-1]
        params = int(latest.get("params", 0) or 0)
        body = latest.get("parameter_breakdown")
        body_params = None
        if isinstance(body, dict):
            body_params = body.get("body")
        lines.append(f"Model params: {params:,}")
        if body_params is not None:
            lines.append(f"Body params: {int(body_params):,}")
        lines.append(
            "Config: "
            f"dim={latest.get('dim')}, heads={latest.get('n_heads')}, "
            f"prelude={latest.get('prelude_layers')}, coda={latest.get('coda_layers')}, "
            f"ffn={latest.get('ffn_hidden_dim')}, loops={latest.get('train_loops')}"
        )
        lines.append(
            "Epoch plan: "
            f"target={latest.get('target_epochs')}, "
            f"updates/epoch={latest.get('updates_per_epoch')}, "
            f"batches/epoch={latest.get('train_batches_per_epoch')}"
        )
    train = df[df["event"] == "train"] if "event" in df.columns else pd.DataFrame()
    if not train.empty:
        latest = train.iloc[-1]
        lines.append(
            "Latest train: "
            f"step={latest.get('step')}, loss={latest.get('loss')}, "
            f"lr={latest.get('lr')}, tok/s={latest.get('tok_s')}, "
            f"epochs={latest.get('epochs_seen')}/{latest.get('target_epochs')}"
        )
    eval_rows = df[df["event"] == "eval"] if "event" in df.columns else pd.DataFrame()
    if not eval_rows.empty:
        latest = eval_rows.iloc[-1]
        lines.append(
            "Latest eval: "
            f"step={latest.get('step')}, loss={latest.get('eval_loss')}, "
            f"ppl={latest.get('eval_ppl')}"
        )
    exact = df[df["event"] == "exact_eval"] if "event" in df.columns else pd.DataFrame()
    if not exact.empty:
        latest = exact.iloc[-1]
        lines.append(
            "Latest exact eval: "
            f"exact={latest.get('exact_exact_match')}, "
            f"EOS={latest.get('exact_eos_rate')}, "
            f"hit-max={latest.get('exact_hit_max_new_tokens_rate')}"
        )
    return "\n\n".join(lines)
