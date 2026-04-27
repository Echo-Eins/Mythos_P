# No-ACT Prototype Accuracy Audit

This audit covers the fixed-loop dense prototype:

Embedding -> prelude dense blocks -> recurrent dense block with LTI injection -> coda dense blocks -> Ada-capable RMSNorm -> tied LM head.

## Fixed In This Pass

- Response-only supervised objective is now the default. Prompt/problem tokens are masked with `-100`; loss trains the solution and final answer tokens.
- Long samples default to `drop` instead of truncating and forcing a fake EOS. `truncate` is still available, but it no longer rewrites the final token to EOS.
- Multi-sample packing is disabled by default. A formatted problem/solution pair gets its own causal chunk unless `--pack-samples` is explicitly passed.
- Train and eval losses are weighted by the number of supervised tokens, not by microbatch count.
- Startup logs now include data filtering stats, loss target, long-sample policy, seed, sequence length, effective batch chunks, and scheduler settings.
- LTI injection now uses the stable EMA-style update `A*h + (1-A)*(B_e*e + B_delta*delta)` instead of an unconstrained `A*h + B*e + delta`.
- Recurrent-depth norms can use loop-conditioned AdaRMSNorm; train/eval emit structured `metrics.jsonl` events for the Web UI.

## Highest-Impact Remaining Limits

- The old 512/1024-dim runs were much smaller than the headline parameter count suggested. With a Qwen-sized vocabulary and tied embeddings, many parameters are in embeddings; the useful transformer/recurrent body must be tracked separately.
- Validation still reports response-token perplexity, not exact-answer accuracy. It can improve while final numeric/symbolic answers remain wrong.
- The training format is plain `Problem:/Solution:/Final answer:` text. Generation must use the same format or quality will look worse than the validation loss implies.
- The dataset solution selector trusts the top-level `solution` field before verified generated candidates. If that field is noisy in the local OpenR1 copy, the model learns noisy reasoning.
- Random validation split is deterministic by seed, but it is not a difficulty/source-stratified benchmark and can leak near-duplicate problem styles.
- If `--pack-samples` is enabled, packed chunks can contain multiple samples separated only by EOS. This is efficient, but attention is not block-diagonal, so answer tokens can attend to previous examples in the same chunk.

## Architecture Limits

- No-ACT fixed recurrence is the right baseline, but one shared recurrent block plus a sinusoidal loop index has limited per-depth specialization.
- The prototype has no per-loop adapters, so loop 1 and loop 4 use the same parameters except for loop-index bias and changing hidden state.
- LTI now gates both input and transformer delta through `(1-A)`, but the nonlinear delta can still dominate if its effective gain grows; watch `lti_effective_delta_abs_max` and loop RMS.
- No MoE means no cheap capacity expansion. This is a capacity limit, not a correctness bug in the first dense baseline.
- No KV-cache does not hurt training accuracy, but it makes generation slow and restricts practical evaluation throughput.

## Training Limits

- `max_seq_len=1024` will drop long reasoning traces under the new default. That is better than training on false endings, but it biases the first run toward shorter/easier solutions.
- Batch size 2 with grad accumulation 8 is only 16 chunks per update. With response-only loss, supervised token count can still vary a lot across updates.
- A 2000-step run is a smoke run, not enough to judge ceiling accuracy.
- No dropout is acceptable for the first underfit run, but longer runs need train/val gap monitoring before increasing capacity.

## Next Checks

- Run no-ACT again with the corrected objective and compare response-token PPL against the old all-token PPL only as a separate metric, not directly.
- Evaluate loop depth from the same checkpoint: 1, 2, 4, 6, and 8 loops.
- Add exact-answer generation eval before treating low PPL as proof of reasoning quality.
- If the new run is stable, test loop-depth overrides from the same checkpoint before adding MoE/MLA.

## Suggested No-ACT Rerun

```bash
python training/train_dense_openr1.py train \
  --dataset-path data/openr1_math_220k/hf_default_train \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --max-seq-len 1024 \
  --loss-on response \
  --long-sample-policy drop \
  --min-response-tokens 16 \
  --dim 1536 \
  --n-heads 24 \
  --prelude-layers 2 \
  --coda-layers 1 \
  --ffn-hidden-dim 4352 \
  --max-loop-iters 4 \
  --train-loops 4 \
  --ada-norm \
  --lti-init-log-dt -2.0 \
  --lti-init-input-gain 1.0 \
  --lti-init-delta-gain 0.35 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-epochs 1.0 \
  --eval-every 500 \
  --save-every 1000 \
  --eval-batches 50 \
  --lr 2e-4 \
  --lr-schedule cosine \
  --warmup-steps 100 \
  --min-lr-ratio 0.1 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --dtype bf16 \
  --device cuda \
  --num-workers 2 \
  --log-every 10 \
  --out-dir runs/dense_openr1_no_act_1536_body117 \
  --metrics-jsonl runs/dense_openr1_no_act_1536_body117/metrics.jsonl \
  2>&1 | tee logs/train_no_act_1536_body117.log
```

## Suggested Exact-Answer Eval

```bash
python training/train_dense_openr1.py exact-eval \
  --checkpoint runs/dense_openr1_no_act_1536_body117/final.pt \
  --dataset-path data/openr1_math_220k/hf_default_train \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --max-samples 100 \
  --seed 2026 \
  --max-new-tokens 512 \
  --n-loops 4 \
  --device cuda \
  --dtype bf16 \
  --predictions-path runs/dense_openr1_no_act_1536_body117/exact_eval_100.jsonl \
  --metrics-jsonl runs/dense_openr1_no_act_1536_body117/metrics.jsonl \
  2>&1 | tee logs/exact_eval_no_act_1536_100.log
```

## Suggested Web UI

```bash
python mythos_gui/app.py
```

Then open `http://localhost:7860` and point the Live Metrics tab at
`runs/dense_openr1_no_act_1536_body117/metrics.jsonl`.
