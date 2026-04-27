# OpenMythos Dense RDT Prototype Plan

## Goal

Build a real causal language-model generator, not a shape-only mock model.

The first prototype must train on `open-r1/OpenR1-Math-220k`, produce logits for
next-token prediction, save/load checkpoints, and generate autoregressive text.
It must preserve the core architectural idea:

```text
Embedding -> Prelude dense blocks -> Recurrent dense block with LTI injection -> Coda dense blocks -> LM head
```

The first prototype deliberately excludes MoE, MLA, and KV cache from the model
path. Those modules stay in `open_mythos/modules.py` for review and later
integration. ACT is now included as an optional recurrent-depth mode, disabled
by default for clean fixed-loop ablations.

## Dataset

- Hugging Face dataset: `open-r1/OpenR1-Math-220k`
- Initial subset: `default`
- Split: `train`
- Format fields to use first:
  - `problem`
  - `solution`
  - `answer`
  - optionally `messages` after inspecting formatting quality
- Training format:

```text
Problem:
{problem}

Solution:
{solution}

Final answer:
{answer}
```

The first run should filter very long samples by token count to keep memory
predictable.

## Prototype Functionality

- Trainable causal LM with cross-entropy next-token loss.
- Full autoregressive `generate()` without KV cache.
- Prompt formatting for math problems.
- Tokenizer adapter around a Hugging Face tokenizer available on the Spark box.
- Checkpoint save/load:
  - model weights
  - optimizer state
  - scheduler state
  - config
  - tokenizer id
  - training step
- Validation loop with loss/perplexity.
- Sampling options:
  - greedy
  - temperature
  - top-k
  - top-p
  - max new tokens
- Loop-depth controls:
  - train loop depth
  - eval/generation loop depth override
- Optional ACT controls:
  - enable/disable ACT
  - halting threshold
  - ponder/compute loss weight
- Basic instrumentation:
  - gradient norm
  - recurrent hidden-state RMS per loop
  - LTI `A().max()`, `A().min()`
  - LTI `B().abs().max()`
  - ACT expected steps / hard steps / halt fraction when enabled
  - tokens/sec
  - GPU memory stats

## Model Architecture

### Dense Prototype

Use only modules that are easy to verify:

- `RMSNorm`
- `RotaryEmbedding`
- `DenseCausalSelfAttention`
- `SwiGLUFFN`
- `DenseTransformerBlock`
- `LTIInjection`
- `DenseRecurrentBlock`
- optional `ACTHalting` + `ACTAccumulator`

### Top-Level Model

Suggested class:

```python
class OpenMythosDenseLM(nn.Module):
    embed
    prelude: ModuleList[DenseTransformerBlock]
    recurrent: DenseRecurrentBlock
    coda: ModuleList[DenseTransformerBlock]
    norm
    lm_head tied to embed
```

Forward:

```python
x = embed(input_ids)
for block in prelude:
    x = block(x)
e = x
x = recurrent(x, e, n_loops=n_loops, cache=None)
for block in coda:
    x = block(x)
logits = lm_head(norm(x))
```

Generation initially recomputes the full growing context every token. This is
slower than cached decoding but much easier to prove correct.

With ACT enabled, the recurrent block computes each loop normally, predicts a
per-token halting probability after the LTI update, accumulates an ACT-weighted
hidden state, and forces the remaining probability mass on the final allowed
loop. The LM loss becomes:

```text
loss = cross_entropy + act_ponder_weight * expected_act_steps
```

## Initial Config Ranges

Pick final numbers only after a short memory probe on the Spark box.

Tiny smoke:

- dim: 256
- heads: 4
- prelude: 1
- recurrent loops: 2-4
- coda: 1
- context: 512-1024

First useful run:

- dim: 512-768
- heads: 8-12
- prelude: 1-2
- recurrent loops: 4-8
- coda: 1-2
- context: 1024-2048

Do not start with 1B. First prove loss decrease and generation quality on a
small dense recurrent model.

## Training Plan

1. Download dataset locally.
2. Build tokenized samples with fixed max length.
3. Run one overfit test on 16-64 samples.
4. Run tiny training for 500-2000 steps.
5. Compare:
   - standard dense transformer with same rough params
   - RDT with `n_loops=1`
   - RDT with `n_loops=2/4/8`
6. Generate math solutions from held-out prompts.
7. Compare fixed-loop training with optional ACT enabled.
8. Only after that consider KV cache, MLA, or MoE.

## Verification Gates

- Forward logits shape is correct.
- Loss decreases on a tiny overfit set.
- `generate()` produces non-empty text and respects EOS/max length.
- LTI `A` remains in `(0, 1)`.
- Hidden-state RMS does not explode across loops.
- With ACT enabled, ACT weights sum to 1 by construction and expected steps are logged.
- Training can resume from checkpoint with matching loss trajectory.
- Increasing eval loops changes outputs without causing NaNs.

## Excluded From First Prototype

- MoE: excluded until dense recurrence is validated.
- MLA: excluded until dense/GQA path and generation are correct.
- KV cache: excluded until non-cached generation is correct.
- 1M context claims: excluded.
