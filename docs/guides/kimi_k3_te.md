# Kimi-K3 with TE MoEBlock

This integration is based on Google's Kimi-K3 stack through PR #6178,
`f9cfba05aa09885417e6b107fd3a90d74d339956`, which includes #5550 and adds
scanned decoder layers. It requires the TE changes in
[pggPL/TransformerEngine#73](https://github.com/pggPL/TransformerEngine/pull/73)
and [#74](https://github.com/pggPL/TransformerEngine/pull/74); the head of #74
contains both changes.

Enable `te_moe_block=true` with the existing `model_name=kimi-k3`. The Google
KDA, gated NoPE MLA, Attention Residuals, dense/shared SiTU MLPs, latent
projections and post-combine RMSNorm remain unchanged. TE computes the routing,
EP dispatch/combine, grouped expert GEMMs and their backward pass.

## Model contract

- The router reads the original hidden state (7168 at production width) in
  FP32; experts read the separately projected latent (3584). Router weights
  retain the full hidden width.
- Routed experts use the actual two-branch SiTU: `4*tanh(gate/4)*sigmoid(gate)`
  times `25*tanh(up/25)`, evaluated in FP32 before casting back. There is no
  SwiGLU replacement.
- Unbiased sigmoid scores are normalized over the selected experts. The
  correction bias affects selection only. Weights are applied after the expert
  down projection, before combine.
- `prefuse_moe_weights=false` preserves Google's separate `wi_0` and `wi_1`
  parameter names and shapes. The adapter concatenates their values for TE;
  gradients return through the concatenate. Checkpoint round-trip execution
  remains to be validated.
- With `quantization=''`, only the routed experts create TE quantizers.
  `te_gmm_quantization=te_no_quant` selects the BF16 baseline; `te_mxfp8` is an
  optional lower-precision execution mode.

Older TE versions fail with an actionable error when separate router inputs or
two-branch SiTU are unavailable. The adapter never substitutes an approximate
model or silently falls back to native routing.

The exact contract here is the architecture and forward/gradient mathematics of
the Google baseline, subject to the selected numerical precision. This change
does not implement missing parts of the published training recipe. In
particular, Google's static correction bias remains static; quantile-balancing
updates and load-balance auxiliary loss propagation are not added. Google's
packing, autoregressive decode, context/pipeline-parallel limitations also
remain. Full official-checkpoint and full-model validation are pending.

## Prepared runs

Inside an installed MaxText environment with the head of TE #74 and
`test-maxtext.sh` on PATH:

```bash
bash tests/end_to_end/gpu/kimi/run_te_exact.sh
# 4 GPUs, EP=2/FSDP=2, 4 layers, 32 experts, top-16, production widths,
# sequence 1024, batch 1/GPU, 20 BF16 steps on synthetic data.

GPUS=2 MAX_TARGET_LENGTH=128 STEPS=5 \
  bash tests/end_to_end/gpu/kimi/run_te_exact.sh

MOE_PARITY=1 bash tests/end_to_end/gpu/kimi/run_te_exact.sh

SCAN_LAYERS=true LAYERS=13 \
  bash tests/end_to_end/gpu/kimi/run_te_exact.sh
```

The default is a reduced training slice with the same building blocks, not a
full-model memory-fit claim. `LAYERS=93 EXPERTS=896` selects the released layer
layout and expert count; the launcher derives the MLA indices from the Google
layout. Such a run requires a separately planned hardware allocation.

`DRY_RUN=1` prints the command. `OUTPUT`, `COORDINATOR_PORT`, `BATCH_PER_GPU`,
`FULL_ATTN_LAYERS`, `TE_GMM_QUANTIZATION` and `EXTRA_MAXTEXT_ARGS` provide
overrides. For smaller component validation, `MOE_PARITY=1` uses a 256-wide
hidden state, 128-wide latent and 8 experts while retaining full-width routing,
two-branch SiTU, RMSNorm, latent projections and a shared SiTU expert.

Worst-case receive capacity (`ragged_buffer_factor<=0`) is required because the
Google latent block does not propagate TE's overflow metadata to the trainer.
This keeps the path dropless without adding an architectural change.

## Validation status

Execution is deferred at the user's request. No model, numerical test, build,
GPU allocation or CI run has been performed for this integration. Source and
shell syntax, formatting and diff checks are recorded separately.

The prepared component check compares BF16 outputs and every parameter/input
gradient against a native oracle with nonzero selection bias and scaling. It
also checks the full-width router and split expert parameter storage. The
existing TE tests cover the separate-input and SiTU primitives; all runtime
checks still need to be run. Scanned/unscanned model parity, rematerialization,
MXFP8 and checkpoint continuation remain follow-up validation.
