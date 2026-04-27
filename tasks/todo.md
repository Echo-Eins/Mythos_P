# Module Extraction Plan

- [x] Read project rules in `AGENTS.md`.
- [x] Review existing project docs/specs relevant to model modules.
- [x] Update `lessons.md` after the user correction.
- [x] Add corrected MLA implementation to `open_mythos/modules.py`.
- [x] Verify syntax without local torch runtime.
- [x] Review diff for obvious shape/cache/masking issues.

## Review

- Added MLA as a reference module, not part of the first dense prototype.
- Corrected original MLA cache semantics by storing `k_rope` as one shared positional head instead of duplicating it across all query heads.
- Updated cached attention masking to use absolute query/key positions; this also fixes the dense attention helper for offset chunks without cache.
- Verified syntax with `py_compile`; runtime tensor tests remain blocked on this machine because local torch is intentionally unavailable.

# Dense Prototype Implementation Plan

- [x] Implement `OpenMythosDenseLM` with tied LM head and non-cached generation.
- [x] Implement OpenR1 dataset formatting/tokenization/packing for training.
- [x] Implement train/eval loop with checkpoint save/load.
- [x] Implement standalone generation mode from checkpoint.
- [x] Verify Python syntax locally without importing torch runtime.
- [x] Review implementation for scope exclusions: no MoE, MLA, KV-cache, 1M context.

## Dense Prototype Review

- Implemented the model path in `open_mythos/dense_lm.py`.
- Implemented OpenR1 train/eval/generate CLI in `training/train_dense_openr1.py`.
- Added Spark-side smoke tests in `tests/test_dense_lm.py`.
- Exported dense prototype classes from `open_mythos/__init__.py`.
- Training keeps model parameters in fp32 and uses CUDA autocast for bf16/fp16 compute.
- Verified syntax with local `py_compile`; runtime tensor tests remain for the Spark environment because local torch is intentionally unavailable.
- Scope check: prototype path imports no MoE/MLA/cache modules and passes `cache=None` through dense blocks.

# ACT Prototype Integration Plan

- [x] Re-check ACT placement in the recurrent architecture.
- [x] Add optional ACT halting to `DenseRecurrentCore`.
- [x] Add ACT ponder/compute loss to the LM objective.
- [x] Add train CLI flags and logging for ACT metrics.
- [x] Add Spark-side ACT smoke test.
- [x] Verify Python syntax locally without importing torch runtime.

## ACT Integration Review

- ACT is optional via `DenseLMConfig.use_act` / `--use-act`; fixed-loop recurrence remains the baseline.
- ACT predicts per-token halting after each LTI recurrent update and returns an ACT-weighted hidden state.
- The final loop forces the remaining ACT mass, so every token contributes total weight 1.
- `act_ponder_loss` is an expected-depth regularizer and is added to LM loss as `act_ponder_weight * act_ponder_loss`.
- Training logs ACT expected steps, hard steps, halt fraction, halting probability, LM loss, ACT loss, and total loss.

# Architecture Sanity Review

- [x] Separate ACT halting normalization from recurrent block normalization.
- [x] Keep eval `loss` as total objective but compute validation `ppl` from pure LM cross-entropy.

## Sanity Review Notes

- ACT is mathematically valid as an expected-depth halting mechanism, but it is still an ablation feature rather than a guaranteed win.
- The LTI linear part is stable by construction (`A` in `(0, 1)`), while the nonlinear transformer delta still needs RMS/gradient monitoring during training.
- The prototype is a complete causal LM generator, but it is not yet a production inference model because KV-cache and optimized per-token ACT scheduling remain excluded.

# Training Logging Fix

- [x] Make training/eval/generation prints line-buffered and flushed so `tee` receives logs every `--log-every` step.

# Long-Run Scheduler And ACT Usability

- [x] Replace the zero-ending HF cosine scheduler with a local scheduler that honors `--min-lr-ratio`.
- [x] Add `--decay-steps` so long runs can decay to a floor and then continue training at that floor.
- [x] Add `--lr-schedule` with `cosine`, `linear`, and `constant` modes.
- [x] Sync optimizer LR to the new scheduler after resume so old zero-LR checkpoints can continue.
- [x] Add `act_min_steps` / `--act-min-steps` to stop ACT from collapsing useful recurrent depth too early.

# No-ACT Accuracy Audit

- [x] Re-check data formatting, label construction, and packing.
- [x] Fix supervised objective so the default loss trains the response, not prompt copying.
- [x] Stop treating truncated long reasoning traces as complete EOS-terminated solutions.
- [x] Improve startup/log metrics that affect run comparability.
- [x] Record remaining architecture/training limitations for the next prototype pass.

## No-ACT Accuracy Audit Review

- Added `tasks/no_act_accuracy_audit.md` with fixed issues, remaining accuracy limits, and next checks.
- Default training objective now targets response tokens instead of prompt copying.
- Long examples are dropped by default rather than converted into false completed solutions.
- Packing keeps examples intact when they fit in the configured context.
- Loss and perplexity are now weighted by supervised tokens, which matters after prompt masking.

# Exact Answer Evaluation And 1024-Dim Baseline

- [x] Raise the dense prototype default width to `dim=1024`.
- [x] Add dataset-driven exact-answer generation evaluation with the same prompt format as training.
- [x] Add answer extraction and normalization for OpenR1-style final answers.
- [x] Add eval logging/prediction output for debugging wrong answers.
- [x] Verify syntax locally without importing torch runtime.

## Exact Eval Review

- Added `exact-eval` subcommand to `training/train_dense_openr1.py`.
- Exact eval prompts with `Problem:\n{problem}\n\nSolution:\n`, matching the supervised generation prefix and excluding the target answer.
- The evaluator extracts `\boxed{...}`, `Final answer:`, `####`, or the final non-empty line, then compares normalized exact answers.
- Predictions can be written to JSONL with generated text, target, prediction, normalized forms, and prompt token count.
