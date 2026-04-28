#!/usr/bin/env python3
"""Gradio Web UI for monitoring OpenMythos dense prototype runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mythos_gui.live_monitor import (  # noqa: E402
    empty_figure,
    events_to_frame,
    generation_dashboard,
    load_events,
    recurrent_dashboard,
    summary_markdown,
    training_dashboard,
)


def _tail_table(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    columns = [
        col
        for col in (
            "event",
            "step",
            "loss",
            "lm_loss",
            "eval_loss",
            "eval_lm_loss",
            "eval_ppl",
            "lr",
            "grad_norm",
            "tok_s",
            "input_tok_s",
            "padded_tok_s",
            "padding_efficiency",
            "real_seq_len_mean",
            "padded_seq_len_mean",
            "epochs_seen",
            "epoch_index",
            "epoch_number",
            "current_epoch",
            "epoch_progress",
            "target_epochs",
            "lti_A_min",
            "lti_A_max",
            "lti_effective_input_abs_max",
            "lti_effective_delta_abs_max",
            "act_expected_steps",
            "exact_exact_match",
            "exact_eos_rate",
            "exact_hit_max_new_tokens_rate",
        )
        if col in df.columns
    ]
    if not columns:
        return df.tail(limit)
    return df[columns].tail(limit)


def refresh_metrics(metrics_path: str) -> tuple[str, go.Figure, go.Figure, go.Figure, pd.DataFrame]:
    events = load_events(metrics_path)
    df = events_to_frame(events)
    return (
        summary_markdown(df),
        training_dashboard(df),
        recurrent_dashboard(df),
        generation_dashboard(df),
        _tail_table(df),
    )


def load_predictions(predictions_path: str) -> tuple[str, go.Figure, pd.DataFrame]:
    if not predictions_path.strip():
        return "No predictions file selected.", empty_figure("Prediction diagnostics"), pd.DataFrame()
    path = Path(predictions_path).expanduser()
    if not path.exists():
        return f"File does not exist: `{path}`", empty_figure("Prediction diagnostics"), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    if not rows:
        return "No prediction rows found.", empty_figure("Prediction diagnostics"), pd.DataFrame()

    df = pd.DataFrame(rows)

    def numeric_column(name: str, default: float = 0.0) -> pd.Series:
        if name not in df.columns:
            return pd.Series([default] * len(df))
        return pd.to_numeric(df[name], errors="coerce").fillna(default)

    exact = numeric_column("exact")
    generated = numeric_column("generated_tokens")
    eos = numeric_column("eos_emitted")
    hit_max = numeric_column("hit_max_new_tokens")
    repeat = numeric_column("repeat_4gram_ratio")

    summary = (
        f"Rows: {len(df):,}\n\n"
        f"Exact match: {exact.mean():.4f}\n\n"
        f"EOS rate: {eos.mean():.4f}\n\n"
        f"Hit max-new-tokens rate: {hit_max.mean():.4f}\n\n"
        f"Avg generated tokens: {generated.mean():.2f}\n\n"
        f"Avg repeated 4-gram ratio: {repeat.mean():.4f}"
    )
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=generated, name="generated tokens", nbinsx=40))
    fig.add_trace(go.Histogram(x=repeat, name="repeat 4-gram ratio", nbinsx=40, opacity=0.65))
    fig.update_layout(
        title="Prediction diagnostics",
        template="plotly_white",
        barmode="overlay",
        height=420,
        margin=dict(l=48, r=24, t=60, b=40),
    )

    visible_cols = [
        col
        for col in (
            "row_idx",
            "exact",
            "target",
            "prediction",
            "prompt_tokens",
            "generated_tokens",
            "eos_emitted",
            "hit_max_new_tokens",
            "repeat_4gram_ratio",
            "unique_token_ratio",
            "problem",
            "generated_text",
        )
        if col in df.columns
    ]
    return summary, fig, df[visible_cols].head(500)


def _safe_decode(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _select_best_solution(row: dict[str, Any]) -> str:
    solution = row.get("solution")
    if isinstance(solution, str) and solution.strip():
        return solution.strip()

    generations = row.get("generations")
    complete = row.get("is_reasoning_complete")
    verified = row.get("correctness_math_verify")
    if isinstance(generations, list):
        for idx, candidate in enumerate(generations):
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            ok_complete = not isinstance(complete, list) or idx >= len(complete) or bool(complete[idx])
            ok_verified = not isinstance(verified, list) or idx >= len(verified) or bool(verified[idx])
            if ok_complete and ok_verified:
                return candidate.strip()
        for candidate in generations:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _format_openr1_parts(row: dict[str, Any]) -> tuple[str, str]:
    problem = str(row.get("problem") or "").strip()
    solution = _select_best_solution(row)
    answer = str(row.get("answer") or "").strip()
    if not problem or not solution:
        return "", ""
    prompt = f"Problem:\n{problem}\n\nSolution:\n"
    response = solution
    if answer:
        response += f"\n\nFinal answer:\n{answer}"
    return prompt, response.strip()


def inspect_masks(
    dataset_path: str,
    tokenizer_id: str,
    sample_index: int,
    max_seq_len: int,
    loss_on: str,
    long_sample_policy: str,
    min_response_tokens: int,
    show_tokens: int,
) -> tuple[str, go.Figure, go.Figure, pd.DataFrame]:
    if not dataset_path.strip() or not tokenizer_id.strip():
        return (
            "Dataset path and tokenizer are required.",
            empty_figure("Causal attention mask"),
            empty_figure("Label mask"),
            pd.DataFrame(),
        )

    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(str(Path(dataset_path).expanduser()))
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    if len(ds) == 0:
        return "Dataset is empty.", empty_figure("Causal attention mask"), empty_figure("Label mask"), pd.DataFrame()

    idx = int(sample_index) % len(ds)
    row = ds[idx]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt, response = _format_openr1_parts(row)
    if not prompt or not response:
        return "Selected row has no valid prompt/response.", empty_figure("Causal attention mask"), empty_figure("Label mask"), pd.DataFrame()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)
    if len(response_ids) < min_response_tokens:
        return "Selected row is below min_response_tokens.", empty_figure("Causal attention mask"), empty_figure("Label mask"), pd.DataFrame()

    ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    label_mask = [1] * len(ids) if loss_on == "all" else response_mask
    status = "kept"
    if len(ids) > max_seq_len + 1:
        if long_sample_policy == "drop":
            status = "would be dropped"
        else:
            ids = ids[: max_seq_len + 1]
            label_mask = label_mask[: max_seq_len + 1]
            response_mask = response_mask[: max_seq_len + 1]
            status = "truncated"

    input_ids = ids[:-1][:max_seq_len]
    target_ids = ids[1 : len(input_ids) + 1]
    supervised = label_mask[1 : len(input_ids) + 1]
    show = max(1, min(int(show_tokens), len(input_ids)))

    causal = [[1 if key <= query else 0 for key in range(show)] for query in range(show)]
    causal_fig = go.Figure(data=go.Heatmap(z=causal, colorscale="Blues", showscale=True))
    causal_fig.update_layout(
        title="Causal attention mask",
        xaxis_title="Key position",
        yaxis_title="Query position",
        template="plotly_white",
        height=520,
        margin=dict(l=56, r=24, t=56, b=48),
    )

    label_fig = go.Figure(
        data=go.Heatmap(
            z=[supervised[:show], response_mask[:show]],
            y=["loss target", "response token"],
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            showscale=True,
        )
    )
    label_fig.update_layout(
        title="Supervision mask",
        xaxis_title="Input position",
        template="plotly_white",
        height=260,
        margin=dict(l=84, r=24, t=56, b=48),
    )

    table = pd.DataFrame(
        {
            "pos": list(range(show)),
            "input_id": input_ids[:show],
            "target_id": target_ids[:show],
            "token": [_safe_decode(tokenizer, token_id) for token_id in input_ids[:show]],
            "target": [_safe_decode(tokenizer, token_id) for token_id in target_ids[:show]],
            "loss_target": supervised[:show],
        }
    )
    summary = (
        f"Sample index: {idx}\n\n"
        f"Status under current policy: {status}\n\n"
        f"Prompt tokens: {len(prompt_ids):,}\n\n"
        f"Response tokens: {len(response_ids):,}\n\n"
        f"Sequence tokens: {len(ids):,}\n\n"
        f"Supervised shifted tokens: {sum(supervised):,}"
    )
    return summary, causal_fig, label_fig, table


def checkpoint_summary(checkpoint_path: str) -> str:
    if not checkpoint_path.strip():
        return "No checkpoint selected."
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        return f"Checkpoint does not exist: `{path}`"

    import torch

    from open_mythos.dense_lm import OpenMythosDenseLM, dense_lm_config_from_dict

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dense_lm_config_from_dict(ckpt["config"])
    model = OpenMythosDenseLM(cfg)
    breakdown = model.parameter_breakdown()
    return (
        f"Checkpoint: `{path}`\n\n"
        f"Step: {ckpt.get('step')}\n\n"
        f"Tokenizer: `{ckpt.get('tokenizer')}`\n\n"
        f"Total params: {breakdown['total']:,}\n\n"
        f"Body params: {breakdown['body']:,}\n\n"
        f"Config: dim={cfg.dim}, heads={cfg.n_heads}, "
        f"prelude={cfg.prelude_layers}, coda={cfg.coda_layers}, "
        f"ffn={cfg.resolved_ffn_hidden_dim()}, loops={cfg.max_loop_iters}, "
        f"ada_norm={cfg.use_ada_norm}, act={cfg.use_act}"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="OpenMythos Monitor") as demo:
        gr.Markdown("# OpenMythos Monitor")

        with gr.Tab("Live Metrics"):
            metrics_path = gr.Textbox(
                label="metrics.jsonl or raw log path",
                value="runs/dense_openr1_no_act_1536_body117/metrics.jsonl",
            )
            with gr.Row():
                refresh = gr.Button("Refresh", variant="primary")
                auto_refresh = gr.Checkbox(label="Auto refresh", value=True)
            summary = gr.Markdown()
            train_plot = gr.Plot()
            recurrent_plot = gr.Plot()
            generation_plot = gr.Plot()
            event_table = gr.Dataframe(label="Latest metric events", wrap=True)

            refresh.click(
                refresh_metrics,
                inputs=metrics_path,
                outputs=[summary, train_plot, recurrent_plot, generation_plot, event_table],
            )
            timer = gr.Timer(value=5.0, active=True)
            timer.tick(
                lambda path, enabled: refresh_metrics(path)
                if enabled
                else (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                ),
                inputs=[metrics_path, auto_refresh],
                outputs=[summary, train_plot, recurrent_plot, generation_plot, event_table],
            )

        with gr.Tab("Generation Diagnostics"):
            predictions_path = gr.Textbox(label="predictions JSONL", value="")
            load_pred = gr.Button("Load predictions", variant="primary")
            pred_summary = gr.Markdown()
            pred_plot = gr.Plot()
            pred_table = gr.Dataframe(label="Prediction rows", wrap=True)
            load_pred.click(
                load_predictions,
                inputs=predictions_path,
                outputs=[pred_summary, pred_plot, pred_table],
            )

        with gr.Tab("Attention And Label Masks"):
            with gr.Row():
                dataset_path = gr.Textbox(
                    label="OpenR1 dataset path",
                    value="data/openr1_math_220k/hf_default_train",
                )
                tokenizer_id = gr.Textbox(label="Tokenizer", value="Qwen/Qwen2.5-1.5B")
            with gr.Row():
                sample_index = gr.Number(label="Sample index", value=0, precision=0)
                max_seq_len = gr.Number(label="Max seq len", value=1024, precision=0)
                show_tokens = gr.Slider(label="Visible tokens", minimum=16, maximum=256, value=128, step=16)
            with gr.Row():
                loss_on = gr.Radio(["response", "all"], label="Loss target", value="response")
                long_policy = gr.Radio(["drop", "truncate"], label="Long sample policy", value="drop")
                min_response = gr.Number(label="Min response tokens", value=16, precision=0)
            inspect = gr.Button("Inspect sample", variant="primary")
            mask_summary = gr.Markdown()
            causal_plot = gr.Plot()
            label_plot = gr.Plot()
            token_table = gr.Dataframe(label="Visible tokens", wrap=True)
            inspect.click(
                inspect_masks,
                inputs=[
                    dataset_path,
                    tokenizer_id,
                    sample_index,
                    max_seq_len,
                    loss_on,
                    long_policy,
                    min_response,
                    show_tokens,
                ],
                outputs=[mask_summary, causal_plot, label_plot, token_table],
            )

        with gr.Tab("Checkpoint"):
            checkpoint_path = gr.Textbox(label="Checkpoint path", value="")
            load_ckpt = gr.Button("Load summary", variant="primary")
            ckpt_summary = gr.Markdown()
            load_ckpt.click(checkpoint_summary, inputs=checkpoint_path, outputs=ckpt_summary)

        demo.load(
            refresh_metrics,
            inputs=metrics_path,
            outputs=[summary, train_plot, recurrent_plot, generation_plot, event_table],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
