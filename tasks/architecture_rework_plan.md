# Dense Recurrent Architecture Rework Plan

## Scope

This plan verifies `C:\Users\EchoEins\Downloads\Analysis1` against
`logs/28_04_5 no act 1536 117M dynpad b4_4_RB fix` and the active dense path:

- `training/train_dense_openr1.py`
- `open_mythos/dense_lm.py`
- `open_mythos/modules.py`

The goal is not to add more features blindly. The goal is to make the dense
recurrent LM mathematically coherent, measurable, and capable of gaining quality
from recurrent depth.

## Checked Literature

- Geiping et al., 2025, `Scaling up Test-Time Compute with Latent Reasoning`.
  Supports prelude/core/coda, random or learned latent recurrent state, input
  reinjection at every recurrent step, concat adapter for `(h, e)`, random-depth
  training, and truncated backpropagation through recurrent depth.
- Bae et al., 2024, `Relaxed Recursive Transformers`.
  Supports depth-wise LoRA as a compact way to relax strict layer tying.
  The exact Analysis1 name "Mixture-of-Depths LoRA" is not the primary title.
- Dao and Gu, 2024, `Transformers are SSMs`.
  Supports selective/structured SSM design and Mamba-2 efficiency, but does not
  prove that a token-dependent `A` will improve this exact dense recurrent LM.
- Hao et al., 2024, `COCONUT`.
  Supports continuous hidden-state reasoning as a separate training paradigm,
  not as a small patch to the current OpenR1 SFT run.
- Dehghani et al., 2018, `Universal Transformers`.
  Supports recurrent transformer computation and ACT-style per-position
  adaptive halting, but this is not enough to justify enabling ACT before the
  fixed-depth recurrent core is useful.

## Verified Analysis1 Claims

- Training is stable. The run reaches step 5200 with no divergence; eval loss
  reaches `1.4489`, PPL `4.2582` at step 5000.
- Dynamic padding works. Runtime padding efficiency is about `0.98`, while the
  static-padding efficiency of the dataset is about `0.406`.
- The run uses the intended aggressive LTI init: `log_dt=-0.5`, input gain
  `0.3`, delta gain `0.35`.
- `loop_rms` decreases across recurrent steps near the end:
  `[1.750, 1.399, 1.182, 1.058]` at step 5000. This is a real warning that the
  recurrent state is shrinking instead of clearly refining.
- The current recurrent transformer step sees only `h + loop_embedding`.
  `e` enters directly through LTI injection, not through the transformer update.
- `coda_layers=1` is a weak readout after a lossy recurrent state.
- `weight_decay=0.1` is currently applied through one AdamW group to all
  parameters, including embeddings and norms.
- `pin_memory` is already enabled when CUDA is available; this part of
  Analysis1 is not a missing feature.
- No exact-answer or generation diagnostics are present in this log, so the run
  proves teacher-forced response-token PPL, not real answer accuracy.

## Corrections To Analysis1

- Do not blindly restore `base = norm(h_loop + e)`. That reintroduces the old
  double input dominance problem. The correct fix is a learned `(h, e)` mixer
  inside the recurrent step.
- Do not blindly raise `init_input_gain` to `1.0` or remove `(1 - A)` from the
  input path. That changes the LTI equilibrium and may make the recurrence an
  input skip rather than a refiner. Keep LTI stable and fix the recurrent
  contract instead.
- `loop_rms` decay alone does not prove loss of information. It proves scale
  attenuation. The next implementation must log `delta_rms`, drive RMS,
  `h/e` cosine, recurrent output norm, and loop-depth validation.
- Tokenizer/vocab trimming is high-risk because it changes the Qwen tokenizer
  id space and checkpoint compatibility. It is not a phase-1 accuracy fix.
- MXFP4/fp8 training is not an architecture fix. It should wait until bf16
  recurrent-depth quality is real.
- Noise between loops and loop-step conditioning are not must-have. Geiping's
  recurrent-depth ablation reports that similar variants did not help in their
  preliminary experiments.

## Implementation Plan

### Phase 0: Measurement Before More Architecture Changes

- Add exact-answer evaluation to the post-run checklist for every serious run:
  exact match, EOS rate, hit-max-token rate, repetition rate, average generated
  length, and wrong-answer JSONL samples.
- Add loop-depth sweep evaluation for `n_loops in {1, 2, 4, 6, 8}`. A recurrent
  architecture is not validated unless quality improves or at least holds when
  depth increases.
- Extend recurrent stats:
  `delta_rms`, `drive_rms`, `input_drive_rms`, `delta_drive_rms`,
  `h_e_cosine`, `h_delta_cosine`, per-loop norm before and after LTI,
  and coda-input RMS.
- Keep the current checkpoint as a baseline and run exact-eval/loop-sweep before
  training the next architecture. Otherwise improvements are not attributable.

### Phase 1: Fix The Recurrent Contract

- Add `RecurrentInputMixer` in `dense_lm.py`:
  - `concat_project`: `mix = Linear([RMS(h_loop), RMS(e)] -> dim)`.
  - `gated`: `mix = gate * h_loop + (1 - gate) * e`.
  - Default for the main branch: concat projection initialized close to current
    behavior but with a small explicit `e` path, not raw `h + e`.
- Feed the mixed state to the recurrent transformer:
  `base = recurrent_norm(mix, norm_cond)`.
- Keep direct LTI input injection, but do not use it as the only way the
  transformer sees the problem state.
- Add `recurrent_out_norm` before coda.
- Add an output bridge into coda:
  `coda_input = output_mixer([recurrent_out_norm(h), e])`, initialized so the
  recurrent path dominates but `e` is not destroyed by LTI attenuation.
- Raise the recommended accuracy config to `coda_layers=2`. If memory is tight,
  reduce FFN hidden width before reducing coda depth.
- Add config/version fields so old checkpoints fail loudly or load through an
  explicit compatibility path.

### Phase 2: Make Depth Trainable

- Add random-depth training:
  - `--train-loop-schedule fixed|uniform|lognormal-poisson`.
  - `--train-loops-min`, `--train-loops-max`, `--train-loops-mean`.
  - Default experimental branch: log-normal Poisson around mean `4`, max `8`
    or `12`, then scale after a successful short run.
- Add truncated recurrent-depth backprop:
  - `--bptt-depth`.
  - Early loop states are detached; the last `bptt_depth` loops keep gradients.
  - `e` remains injected in the differentiable suffix so the prelude still gets
    useful gradients.
- Add `--recurrent-init encoded|learned|learned-noisy`.
  - Keep `encoded` for compatibility.
  - Test `learned-noisy`: learned initial latent state at eval, small Gaussian
    noise during train only.
- Add loop-depth validation to every eval event. The target signal is not only
  lower loss; it is non-negative depth scaling.

### Phase 3: Fix Optimizer And Data Runtime

- Replace the single AdamW group with parameter groups:
  - decay: attention/FFN/mixer matrix weights.
  - no decay: embeddings, tied LM head, RMS/AdaRMSNorm weights, biases, LTI
    scalars/gains, halting/gating scalars.
- Increase effective batch for serious runs:
  - first target: `batch_size=4`, `grad_accum=16`.
  - only use `32` after confirming wall-clock and LR schedule behavior.
- Use multi-epoch schedules explicitly:
  - either constant LR after warmup for architecture ablations,
  - or cosine with `--decay-steps` equal to total intended update count.
- Add DataLoader knobs:
  - `persistent_workers=True` when `num_workers > 0`.
  - configurable `prefetch_factor`.
  - default serious-run `--num-workers 4`.
- Reduce `--length-bucket-mult` to `32` first. Keep `64` only if throughput
  clearly regresses.
- Keep `torch.compile` as a measured option, not a default, because dynamic
  padding can create shape recompilation unless bucketing is stable.

### Phase 4: Add Capacity Only After Depth Works

- Add depth-wise LoRA adapters if Phase 1/2 shows loop-depth helps but saturates:
  - rank `8` or `16`.
  - zero/no-op initialization.
  - attach first to attention output and FFN down projection.
- Add selective/token-dependent LTI only as a controlled branch:
  - low-rank projection for `A_t`.
  - initialize around current scalar `A`.
  - clamp and log full `A_t` distribution.
- Consider MoDr-style multi-branch recurrence only after depth-wise LoRA proves
  useful. It is a separate architecture branch, not a small cleanup.
- Keep ACT off until fixed-depth recurrence has useful depth scaling. ACT is an
  efficiency/adaptive-compute mechanism, not a repair for a broken recurrent
  operator.

### Phase 5: Explicit Non-Goals For This Pass

- No MoE/MLA/KV-cache until the dense recurrent contract is fixed and exact-eval
  improves.
- No tokenizer remapping/vocab trimming in the main path.
- No MXFP4/fp8 training until bf16 architecture behavior is validated.
- No DEQ/Anderson inference until loop-depth sweeps show a useful fixed point.
- No COCONUT-style objective in this patch. It should be a separate training
  experiment with its own data format and metrics.

## Verification Plan

- Local no-torch verification:
  - `py_compile` for changed Python files.
  - static checks for config/checkpoint compatibility paths.
- Spark runtime smoke tests:
  - forward/backward with fixed loops.
  - forward/backward with random-depth schedule.
  - checkpoint save/load across new config fields.
  - exact-eval on a tiny OpenR1 subset.
  - loop-depth sweep on a tiny OpenR1 subset.
- Short ablation run:
  - 500-1000 update smoke for Phase 1 fixed loops.
  - Compare to current run at matched update count and matched token count.
- Serious ablation run:
  - `max_epochs=2-3`, `grad_accum=16`, coda `2`, fixed loops.
  - Then same budget with random-depth + truncated BPTT.
- Kill criteria:
  - If eval loss is worse by more than 5% at matched token count and exact-eval
    does not improve, revert or reduce the new mixer/output-bridge strength.
  - If `n_loops > train_loops` consistently worsens loss/accuracy, do not add
    LoRA/MoE; fix depth training first.
