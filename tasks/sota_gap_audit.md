# SOTA Gap Audit

This audit separates the current train path from reference modules.

Current training path:

`training/train_dense_openr1.py` -> `open_mythos.dense_lm.OpenMythosDenseLM` -> selected blocks from `open_mythos.modules`.

Reference or older paths:

- `open_mythos.modules`: reviewed building blocks, many not yet integrated into `OpenMythosDenseLM`.
- `open_mythos.main`: older full architecture sketch with MLA/MoE/ACT/cache, not the current training path.
- `open_mythos.moda`: separate MoDA/DeepSeekMoE experiment, not the current training path.

## Immediate Correctness Gaps

- Multi-sample packing without block-diagonal attention was a train/generation mismatch. Fixed by making packing opt-in via `--pack-samples`.
- Exact-answer eval exists, but it is lexical plus simple numeric/fraction normalization, not a symbolic verifier.
- The current dense model has no KV-cache, so generation is correct but inefficient and hard to evaluate at scale.
- The old `open_mythos.main.RecurrentBlock` ACT path can return an underweighted hidden state if the cumulative ACT mass never reaches the threshold before the loop budget ends.
- `open_mythos.moda.DeepSeekGate` stores aux-loss-free routing bias as `nn.Parameter`; unless excluded from the optimizer, this contradicts the intended non-gradient bias update.

## Architecture Gaps Vs Modern Decoder LMs

- The active prototype is dense-only: no MoE, no MLA, no KV-cache, no per-loop adapters, no block-diagonal packed attention.
- The 1024-dim model has ~193M parameters, but about 155M are tied token embeddings with the Qwen vocabulary. The useful transformer/recurrent body is only ~38M unique parameters.
- Only one recurrent block is shared across all loops. Loop index embedding helps, but there are no per-loop LoRA/adapters/router-biases for phase specialization.
- LTI constrains the linear recurrence, but the nonlinear transformer delta has no learned residual gate or depth-scaled initialization.
- RoPE is basic lazy RoPE. There is no long-context scaling, YaRN/NTK scaling, attention sink handling, or context-extension validation.
- Attention uses PyTorch SDPA, which may dispatch efficient kernels, but there is no explicit FlashAttention-2/varlen path and no segment-level reset masks.
- MoE in `modules.py` is a correct reference layer, not a SOTA MoE implementation: no grouped GEMM, capacity management, expert parallelism, communication overlap, or integrated router-bias update.
- MLA in `modules.py` is a corrected reference implementation, not integrated into the dense LM and not optimized for decode throughput.

## Training/Evaluation Gaps

- Training is SFT-only on OpenR1 Math. There is no base pretraining stage, no curriculum, no quality filtering beyond simple field selection, and no DPO/RL/rejection-sampling loop.
- Validation perplexity is response-token teacher-forced loss. It is useful but insufficient for reasoning quality.
- Exact eval samples from the same raw split unless a separate dataset path/split is passed. A held-out benchmark split is still needed.
- There is no symbolic math verifier in the eval loop.
- There is no repetition metric, EOS-rate metric, generated length histogram, or loop-depth exact-eval sweep.
- Checkpointing is basic single-file save. There is no EMA/SWA, sharding, distributed optimizer, or resume safety validation beyond loading states.

## Priority Before Further Scaling

1. Keep no-packing SFT and rerun a short sanity run to verify generation stops improving.
2. Add generation diagnostics: EOS rate, avg length, repeated n-gram ratio, exact match by checkpoint.
3. Add block-diagonal/segment attention if packing is needed for speed.
4. Add a small base-pretraining or LM warmup stage before math SFT, or start from a pretrained backbone if the goal is useful math generation rather than architecture proof.
5. Add per-loop adapter or residual gate only after loop-depth sweeps show whether recurrence is being used.
6. Integrate MoE into the recurrent FFN only after dense no-pack generation is sane.
7. Integrate KV-cache before large exact-eval sweeps.
8. Treat MLA as an inference-efficiency feature after dense/GQA correctness is stable.
