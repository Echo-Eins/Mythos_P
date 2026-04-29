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

# No-ACT 1024 Run Diagnosis

- [x] Parse `logs/27_04_3_no act_1024` startup, train, eval, and exact-eval lines.
- [x] Confirm optimizer stability and falling response-token validation loss.
- [x] Identify sample packing without block-diagonal attention as a train/generation mismatch.
- [x] Disable multi-sample packing by default for supervised math fine-tuning.
- [x] Update diagnosis notes and next-run command.
- [x] Verify syntax locally without importing torch runtime.

## No-ACT 1024 Diagnosis Review

- `logs/27_04_3_no act_1024` shows stable training: eval loss went `3.07 -> 1.62`, PPL `21.60 -> 5.04`.
- Exact-eval failed because generation is repetitive and did not emit EOS in the inspected samples, not because optimizer was dead.
- Root causes to address first: multi-sample causal packing mismatch, too-short 2000-step scratch run, and LR decay reaching the floor while the model was still improving.
- Training now defaults to one sample per causal chunk; `--pack-samples` is opt-in only.

# Full SOTA Gap Audit

- [x] Re-read dense LM, shared modules, training, eval, and tests.
- [x] Compare each implemented mechanism against modern decoder-only/MoE/SFT practice.
- [x] Separate correctness bugs from quality/performance gaps.
- [x] Produce a prioritized action plan before further training.

## Full SOTA Gap Audit Review

- Added `tasks/sota_gap_audit.md`.
- Current train path is a dense baseline, not the full documented Mythos architecture.
- MoE, MLA, KV-cache, and per-loop adapters are not yet integrated into `OpenMythosDenseLM`.
- The next work should focus on generation/objective correctness before adding MoE/MLA.

# Web UI, LTI, AdaNorm, And Body-117 Pass

- [x] Review CERBER `app.py`, `live_monitor.py`, and `inference_diagnostics.py` for reusable monitoring patterns.
- [x] Add structured train/eval/exact-eval JSONL metric events.
- [x] Add explicit epoch accounting and optional `--max-epochs` train target.
- [x] Fix train `lm_loss` logging so it reports the same log-window average as objective loss instead of one noisy microbatch.
- [x] Make epoch numbering explicit: zero-based `epoch_index/current_epoch` plus one-based `epoch_number`.
- [x] Add Mythos-specific Gradio/Plotly Web UI for metrics, recurrent diagnostics, generation diagnostics, and attention/label masks.
- [x] Replace the active LTI update with bounded EMA-style input and delta injection.
- [x] Add loop-conditioned AdaRMSNorm support without adding unused full-dim Ada projections to ordinary blocks.
- [x] Raise the recommended no-ACT body configuration to about 120M body parameters.
- [x] Verify Python syntax locally without importing torch runtime.

## Web UI And LTI Review

- `mythos_gui/app.py` launches a local monitor with live train/eval plots, LTI/loop RMS plots, exact-eval diagnostics, prediction JSONL inspection, OpenR1 causal/label mask inspection, and checkpoint summaries.
- Training writes `metrics.jsonl` by default under `--out-dir`; exact-eval can append diagnostics with `--metrics-jsonl`.
- Training logs `epochs_seen`, `target_epochs`, `current_epoch`, `epoch_progress`, and emits `epoch_end` events when the shuffled train loader is exhausted.
- LTI now logs raw/effective input and delta gains so instability can be diagnosed from graphs instead of raw logs.
- The recommended no-ACT command now uses `dim=1536`, `n_heads=24`, `prelude_layers=2`, `coda_layers=1`, and `ffn_hidden_dim=4352`, giving roughly 120M non-embedding body parameters.

# Dynamic Padding Pass

- [x] Convert non-packed supervised samples from static 1024-token padding to variable-length stored chunks.
- [x] Add an optimized collator that pads each batch to its local maximum length, optionally aligned to a small multiple.
- [x] Add stochastic length-grouped training batches to improve padding efficiency without fully sorting the epoch.
- [x] Preserve correctness for labels, response-only masking, pad masking, and optional sample packing.
- [x] Update defaults/recommended command to `batch-size=4`, `grad-accum=4`, no compile, no curriculum.
- [x] Add local syntax verification and document the expected speedup mechanism.

## Dynamic Padding Review

- Non-packed OpenR1 samples are now stored at their real token length; packed chunks are also left unpadded until collate time.
- `CausalBatchCollator` right-pads each batch to the local maximum sequence length aligned by `--pad-to-multiple`, capped at `--max-seq-len`.
- Training uses shuffled length buckets by default; disable with `--no-length-bucketing` if a fully random batch mix is needed.
- Pad tokens remain future positions for real tokens under causal attention, and all pad labels are forced to `-100`.
- Train logs now include padding efficiency plus real/padded sequence length and throughput metrics so the speedup is visible in the Web UI.
- Syntax verification passed with `py -3 -m py_compile` for the train script, UI monitor, and dynamic padding tests.

# Recurrent LTI Audit Fixes

- [x] Make active LTI defaults less inert for a 4-loop recurrent core.
- [x] Remove double direct input dominance inside the recurrent block path.
- [x] Make AdaRMSNorm loop conditioning state-dependent instead of pure loop-index constant.
- [x] Use per-loop recurrent layer keys for future cache correctness.
- [x] Align loop-index embedding layout with the legacy `main.py` convention.
- [x] Fix reference MoE top-1 router gradient and use explicit `index_add_` accumulation.
- [x] Add ACT remainder to ponder diagnostics/loss.
- [x] Verify syntax and update the recommended train command.

## Recurrent LTI Audit Review

- LTI defaults are now `init_log_dt=-0.5`, `init_input_gain=0.3`, `init_delta_gain=0.35`, so the initial effective delta path is roughly 0.16 instead of 0.04.
- The recurrent block now transforms `h + loop_embedding` and uses the direct LTI path for persistent input injection, avoiding `h + e` plus direct `e` dominance.
- AdaRMSNorm conditioning inside the recurrent core now receives `h_loop[..., :loop_dim]`, not the constant loop embedding difference.
- Recurrent layer keys include `loop_{i}` for future KV-cache correctness.
- Reference MoE top-1 routing keeps gate probability gradients, and routed accumulation uses `index_add_`.
- ACT logs `act_remainder` and uses `hard_steps + remainder` for the ponder objective.
- Syntax verification passed with `py -3 -m py_compile` for model, training, and tests.

# 117M Dense No-ACT Log Comparison

- [x] Read local project rules, lessons, and current prototype/audit docs.
- [x] Parse `logs/27_04 no act 1536_117M` train/eval metrics.
- [x] Compare against earlier no-ACT/ACT runs.
- [x] Decide what signal is needed before testing new modules.
- [x] Add review notes with the final diagnosis.

## 117M Dense No-ACT Log Review

- `logs/27_04 no act 1536_117M` completed one epoch: `epoch_end` at step 5200, with only an 8-chunk grad-accumulation spill into the next epoch.
- This run is not the latest LTI-fixed configuration: it used `lti_init_log_dt=-2.0`, `lti_init_input_gain=1.0`, `lti_init_delta_gain=0.35`.
- Teacher-forced validation improved to the best value so far: eval loss `1.4741`, PPL `4.3672` at step 5000, compared with the previous no-ACT 1024 run ending at loss `1.6165`, PPL `5.0355`.
- At the same 2000-update point, the 117M-body run was worse than the 1024 run (`1.7903` vs `1.6165` eval loss), so the win came from longer training through a full epoch and larger capacity, not faster early convergence.
- Dynamic padding behaved correctly and efficiently: static-padding efficiency was about `0.406`, while runtime padding efficiency stayed around `0.98`.
- LTI remained too inert in this completed run: effective delta only grew from about `0.044` to `0.063`, so the current LTI-fixed run is still the important ablation before adding modules.
- No exact-answer or generation diagnostics are present in this log, so this run proves lower response-token PPL, not answer correctness.

# 117M Dense No-ACT LTI-Fixed Step-2000 Log Review

- [x] Confirm the log is from the latest LTI/double-injection fixes.
- [x] Parse startup, train, eval, LTI, loop RMS, padding, and epoch metrics.
- [x] Compare against the previous 117M dynpad run and 1024 no-ACT baseline at similar steps.
- [x] Record whether the current run is stable enough to continue to exact-eval.

## 117M Dense No-ACT LTI-Fixed Step-2000 Review

- `logs/27_04_4 no act 1536_117M 1536 dynpad b4_3` is from updated code because it logs `lti_tau_min/max`, but it still passes the old LTI flags: `lti_init_log_dt=-2.0`, `lti_init_input_gain=1.0`, `lti_init_delta_gain=0.35`.
- The run reached step 2030, about `0.3904` of epoch 0. There is no epoch completion or exact-eval in this log.
- At step 2000, validation is essentially unchanged vs the previous 117M dynpad run: new eval loss `1.7955`, PPL `6.0223`; old eval loss `1.7903`, PPL `5.9911`.
- Train stability is normal: step-2000 grad norm `0.960`, no NaN/inf/traceback, CUDA memory about `17.39 GB`.
- Dynamic padding remains effective: step-2000 padding efficiency `0.9775`; last-50-step average around `0.978`.
- LTI is only slightly more active than the previous run at the same step: effective delta `0.0566` vs old `0.0555`, and effective input `0.1516` vs old `0.1499`.
- This log does not test the intended aggressive LTI initialization (`-0.5`, `0.3`, `0.35`), so it is not the decisive LTI-fixed ablation.

# Analysis1 Architecture Rework Planning

- [x] Read `Analysis1`, project rules, lessons, and current architecture plans.
- [x] Parse `logs/28_04_5 no act 1536 117M dynpad b4_4_RB fix` and verify Analysis1 metrics.
- [x] Re-read active dense model, recurrent core, LTI injection, optimizer, batching, and eval/generation code.
- [x] Check literature claims against primary sources where the claim affects implementation priority.
- [x] Produce a staged implementation plan focused on accuracy, stability, and efficiency.

## Analysis1 Architecture Rework Review

- Added `tasks/architecture_rework_plan.md`.
- Confirmed the latest run is stable and dynamic padding works, with eval loss `1.4489` and PPL `4.2582` at step 5000.
- Confirmed the active recurrent core still has a structural contract problem: the transformer update sees `h + loop_embedding`, while `e` enters only through LTI injection.
- Confirmed `loop_rms` shrinks across loops, but treated it as scale attenuation rather than proof of lost information until extra recurrent diagnostics are logged.
- Corrected Analysis1 where it suggested directly restoring `h + e`, blindly increasing LTI input gain, treating `pin_memory` as missing, or prioritizing tokenizer trimming/MXFP4 before recurrent-depth correctness.
- The staged plan prioritizes measurement, learned `(h, e)` recurrent mixing, recurrent output normalization/bridge, random-depth training, truncated recurrent-depth BPTT, optimizer param groups, and only then depth-wise LoRA/selective LTI experiments.

# Loop Sweep Test Log Analysis

- [x] Locate `logs/Tests 28_04` files and identify which commands produced them.
- [x] Parse teacher-forced eval metrics by `n_loops`.
- [x] Parse exact-eval/generation diagnostics by `n_loops`.
- [x] Decide whether the current recurrence benefits from deeper inference.
- [x] Record next implementation step from the results.

## Loop Sweep Test Log Review

- Teacher-forced eval confirms trained depth is the sweet spot: loss goes `1.7163` at 1 loop, `1.4948` at 2 loops, best `1.3895` at 4 loops, then worsens to `1.4075` at 6 and `1.4216` at 8.
- This means recurrence is not useless: 4 loops beat 1 loop by about `0.327` loss and 2 loops by about `0.105` loss.
- It also means there is no depth extrapolation yet: extra inference loops beyond the trained depth degrade quality.
- Exact-eval did not run; it crashed before generation because `exact_eval_start` printed a `Path` through raw `json.dumps`.
- Fixed exact-eval JSON serialization in `training/train_dense_openr1.py` and added `n_loops` to final exact-eval metrics.
- Verified syntax with `py -3 -m py_compile training/train_dense_openr1.py`.
- Next step is to rerun exact-eval loop sweep on the same checkpoint, then implement recurrent mixer/output bridge if exact-eval agrees with teacher-forced degradation beyond 4 loops.
