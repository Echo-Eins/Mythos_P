## Module Extraction Scope

- When the user asks to move "all modules" into a shared module file, include every named architectural block even if it is not planned for the first prototype. If a block is risky or not needed immediately, include it as a reviewed/reference implementation and document that it is excluded from the next prototype, rather than omitting it.

## Training Flags Must Be Wired

- When adding a training CLI flag such as `--min-lr-ratio`, verify it changes runtime behavior and appears in startup logging. A declared but unused training flag can silently invalidate comparisons between runs.

## Supervised Math Runs Need Response-Only Loss

- For OpenR1-style SFT, do not optimize the model to predict the problem header and prompt by default. Mask prompt tokens, keep loss on solution/answer tokens, avoid splitting one sample across independent chunks, and do not force EOS onto truncated reasoning traces.
