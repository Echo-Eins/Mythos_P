# No-ACT Prototype Accuracy Audit

This audit covers the fixed-loop dense prototype:

Embedding -> prelude dense blocks -> recurrent dense block with LTI injection -> coda dense blocks -> RMSNorm -> tied LM head.

## Fixed In This Pass

- Response-only supervised objective is now the default. Prompt/problem tokens are masked with `-100`; loss trains the solution and final answer tokens.
- Long samples default to `drop` instead of truncating and forcing a fake EOS. `truncate` is still available, but it no longer rewrites the final token to EOS.
- Packing is sample-aware. A formatted problem/solution pair is not split across independent chunks when it fits in context.
- Train and eval losses are weighted by the number of supervised tokens, not by microbatch count.
- Startup logs now include data filtering stats, loss target, long-sample policy, seed, sequence length, effective batch chunks, and scheduler settings.

## Highest-Impact Remaining Limits

- The 512-dim run is much smaller than the headline parameter count suggests. With a Qwen-sized vocabulary and tied embeddings, most parameters are in embeddings; the actual transformer/recurrent compute body is small for exact math reasoning.
- Validation still reports response-token perplexity, not exact-answer accuracy. It can improve while final numeric/symbolic answers remain wrong.
- The training format is plain `Problem:/Solution:/Final answer:` text. Generation must use the same format or quality will look worse than the validation loss implies.
- The dataset solution selector trusts the top-level `solution` field before verified generated candidates. If that field is noisy in the local OpenR1 copy, the model learns noisy reasoning.
- Random validation split is deterministic by seed, but it is not a difficulty/source-stratified benchmark and can leak near-duplicate problem styles.
- Packed chunks can contain multiple samples separated only by EOS. This is efficient, but attention is not block-diagonal, so answer tokens can attend to previous examples in the same chunk.

## Architecture Limits

- No-ACT fixed recurrence is the right baseline, but one shared recurrent block plus a sinusoidal loop index has limited per-depth specialization.
- The prototype has no per-loop adapters, so loop 1 and loop 4 use the same parameters except for loop-index bias and changing hidden state.
- LTI keeps the linear part stable, but the nonlinear transformer delta is not explicitly gated or residual-scaled beyond normalization and gradient clipping.
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
- If the new run is stable, increase useful body capacity before adding exotic mechanisms: dim 1024, prelude/coda 2, loops 6.

## Suggested No-ACT Rerun

```bash
python training/train_dense_openr1.py train \
  --dataset-path data/openr1_math_220k/hf_default_train \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --max-seq-len 1024 \
  --loss-on response \
  --long-sample-policy drop \
  --min-response-tokens 16 \
  --dim 1024 \
  --n-heads 8 \
  --prelude-layers 1 \
  --coda-layers 1 \
  --max-loop-iters 4 \
  --train-loops 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-steps 2000 \
  --eval-every 200 \
  --save-every 500 \
  --eval-batches 50 \
  --lr 3e-4 \
  --lr-schedule cosine \
  --warmup-steps 100 \
  --min-lr-ratio 0.1 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --dtype bf16 \
  --device cuda \
  --num-workers 2 \
  --log-every 10 \
  --out-dir runs/dense_openr1_no_act_1024_response \
  2>&1 | tee logs/train_no_act_1024_response.log
```

## Suggested Exact-Answer Eval

```bash
python training/train_dense_openr1.py exact-eval \
  --checkpoint runs/dense_openr1_no_act_1024_response/final.pt \
  --dataset-path data/openr1_math_220k/hf_default_train \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --max-samples 100 \
  --seed 2026 \
  --max-new-tokens 512 \
  --n-loops 4 \
  --device cuda \
  --dtype bf16 \
  --predictions-path runs/dense_openr1_no_act_1024_response/exact_eval_100.jsonl \
  2>&1 | tee logs/exact_eval_no_act_1024_100.log
```
