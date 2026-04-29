## Module Extraction Scope

- When the user asks to move "all modules" into a shared module file, include every named architectural block even if it is not planned for the first prototype. If a block is risky or not needed immediately, include it as a reviewed/reference implementation and document that it is excluded from the next prototype, rather than omitting it.

## Training Flags Must Be Wired

- When adding a training CLI flag such as `--min-lr-ratio`, verify it changes runtime behavior and appears in startup logging. A declared but unused training flag can silently invalidate comparisons between runs.

## Supervised Math Runs Need Response-Only Loss

- For OpenR1-style SFT, do not optimize the model to predict the problem header and prompt by default. Mask prompt tokens, keep loss on solution/answer tokens, avoid splitting one sample across independent chunks, and do not force EOS onto truncated reasoning traces.
- Do not mask labels by token id when `pad_token_id == eos_token_id`. Padding labels should be `-100` because the collator writes padded label slots as `-100`, not because every label equal to the pad id is unsupervised. Otherwise EOS is removed from the objective and generation never terminates.

## Do Not Pack SFT Samples Without Attention Reset

- Packing multiple supervised examples into one causal context can make validation perplexity look good while generation fails, because answer tokens attend to previous examples during training but not during standalone generation. Use one sample per causal chunk unless the model supports block-diagonal/reset attention masks.

## Adaptive Layers Must Start As No-Ops

- AdaNorm/adaptive modulation projections should be zero-initialized and protected from generic model initialization. Otherwise an adaptive layer changes the base model at step 0, making ablations against RMSNorm and old checkpoints harder to interpret.

## Monitor Training With Structured Events

- For long Spark runs, write JSONL metric events directly from train/eval/exact-eval instead of relying on raw log scraping. Include body parameter counts, LTI gains, loop RMS, generation EOS/hit-max/repetition stats, and exact-match summaries so the Web UI can diagnose quality without reading `tee` logs manually.
- Train-window metrics must be averaged over the same window and weighting. Do not display one microbatch's `lm_loss` next to a multi-step averaged objective loss; it looks like instability even when validation and averaged loss are smooth.
- Epoch metrics must name their indexing convention explicitly. Use zero-based `epoch_index/current_epoch` for code-like progress and one-based `epoch_number` only for human display.
- Any direct `json.dumps` of CLI payloads must pass through `json_safe` or pre-stringify `Path` values. `append_jsonl` already handles this, but console prints can still crash Spark-side eval before the first sample.

## Dynamic Padding For Causal SFT

- For right-padded causal LM batches, pad keys are future positions for real tokens, so real-token outputs stay correct without a separate padding attention mask. Always force pad labels to `-100`, log padding efficiency, and keep a `--static-padding` escape hatch for debugging fixed-shape behavior.

## Recurrent LTI Defaults Must Match Loop Budget

- For a fixed small recurrent budget such as 4 loops, LTI retention near `A=0.87` makes the delta path too weak at initialization. Check `1-A`, effective input/delta gains, and implied time constant against the actual loop count before treating recurrence as active.
- When loop-depth sweeps show trained depth helps but extrapolated depth degrades, do not only rerun the same architecture. Apply the agreed recurrent contract fixes first: explicit `(h, e)` coupling, normalized coda bridge, and depth-aware diagnostics.

## Avoid Fake Adaptive Conditioning

- AdaNorm conditioning should carry useful state or be removed. A constant loop-index-only condition is mostly a per-loop bias path and can hide that the recurrent state is not being used by the adaptive layer.

## Preflight Before Long Spark Runs

- Before asking for another hour-scale training run, provide a fast `preflight` command that exercises dataset loading, label/EOS checks, model construction, forward, backward, eval, generation, and optional save/load. Bugs that can be caught in one or two batches should not be discovered after a long training launch.
- Optional architecture branches should not allocate inactive trainable modules. If ACT is disabled, do not keep unused ACT parameters in the no-ACT optimizer or parameter count; make compatibility for older checkpoints explicit instead.
- With `torch.compile(..., mode="reduce-overhead")`, treat CUDA Graph forward outputs as short-lived. Clone/detach scalar losses and logging stats immediately after forward, before any later compiled model call can overwrite graph output storage.
- Compile preflight must mimic the real training loop closely enough to cover gradient accumulation. A one-microbatch compile smoke test can pass while a multi-microbatch compiled train step fails; run the preflight with the intended `--grad-accum` and enough `--probe-batches`.
