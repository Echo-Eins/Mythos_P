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
