# Kimi-K3 training with Transformer Engine MoE

This integration combines Google's Kimi-K3 text architecture, tokenizer,
checkpoint conversion, scanned decoder and selectable KDA implementation with
Transformer Engine's routed MoE path and global-step Quantile Balancing (QB).
It remains a draft. A small synthetic BF16 workload has been built, checked
and profiled on four GB200 GPUs. Full-model numerical parity, checkpoint
continuation and memory requirements remain unvalidated.

The Google integration base contains these pinned PR heads:

| PR | Commit | Feature |
| --- | --- | --- |
| #5550 | `06659ec278f8281bcfabf2df5531159f61eb3958` | BF16 decoder |
| #5551 | `9e05bf81431affcca031cf0a2fe6175823ab4ce7` | Kimi tokenizer |
| #5542 | `5679fabe4a45735219b7b98477b2a69d213e5ba3` | MXFP4 conversion / dequantize-on-load |
| #6014 | `7fd77aab5ec0c5bf64a881016b4f63e3dd91fad0` | Optional tokamax KDA |
| #6178 | `a35bcb3fe08d3a3bf4601eb638fb44db3c1e49aa` | Scanned decoder and checkpoint mapping |

## TE dependency

Use the stack [TE #73](https://github.com/pggPL/TransformerEngine/pull/73)
→ [#74](https://github.com/pggPL/TransformerEngine/pull/74)
→ [#75](https://github.com/pggPL/TransformerEngine/pull/75), including token-mask
support. The pinned TE revision is
`ee66c66ec2e2205f59b4a0d7f1f1dbd89dce0130`; it is also recorded in
`tests/end_to_end/gpu/kimi/te_revision.txt`. Build TE JAX from that revision in
an appropriate GPU container before running this code. An older TE installation
fails explicitly when a required capability is missing.

The router reads the full FP32 hidden state, while experts process the latent.
TE applies SiTU with caps 4/25 on both FFN branches and applies routing weights
late. MaxText retains separate `wi_0`/`wi_1` checkpoint parameters and concatenates
them only for the TE call. KDA, gated NoPE MLA, Attention Residuals, latent
projections, post-expert RMSNorm and shared experts follow the Google model.

## Quantile Balancing

`kimi-k3.yml` enables `moe_quantile_balancing=true`. Biases are non-optimizer
`MoEBiasVar` state and remain fixed throughout a full gradient-accumulation step.
Each valid token contributes a sample for every expert. Histograms are collected
without an auxiliary load-balancing loss. The next bias is recovered from the
pooled global-batch quantile and mean-centered, then used on the following step.
Empty batches/layers and rejected skip-step optimizer updates preserve the old
bias. Evaluation does not collect statistics or update biases. Checkpoints
persist the bias, while histogram intermediates are discarded.

The native router uses a JAX/LAX histogram with bounded token chunks. This is a
portable baseline, including TPU; its distributed performance is unmeasured.
The TE path returns per-token-shard histograms. MaxText accumulates them in the
microbatch scan carry and reduces them once after the full step, rather than
stacking a histogram for every microbatch or averaging local quantiles.

`moe_qb_num_bins=1000` controls the histogram resolution. TE also accepts
`moe_qb_histogram_mode=fused_atomic` or `two_kernel`.
`moe_qb_mask_padding=true` excludes tokens with input segment ID zero. When a
mask is supplied, TE computes routing/cutoff and JAX collects the masked histogram
from the saved scores; the no-mask path uses the CUDA histogram kernels directly.
Disabling masking is restricted to synthetic data. QB requires sigmoid FP32
routing, no routing groups, zero `load_balance_loss_weight`, and zero
`routed_bias_update_rate`. Tunix gradient accumulation is not integrated.

## GPU configuration

Select `model_name=kimi-k3 te_moe_block=true`, with:

```text
quantization= te_gmm_quantization=te_no_quant
sparse_matmul=true megablox=false prefuse_moe_weights=false
use_ring_of_experts=false ragged_buffer_factor=0.0
moe_quantile_balancing=true
```

`te_gmm_quantization=te_mxfp8` opts into MXFP8 grouped GEMMs for routed experts.
This is distinct from the released MXFP4 weight format. Imported MXFP4 weights
must be dequantized into dense training parameters for this TE path. Google's
packed MXFP4 runtime remains available for native inference/parity; it is not
trainable and is not native MXFP4 grouped GEMM/QAT support in TE.

`run_te_exact.sh` is a local multi-process launcher. Set `GPUS` explicitly; `EP`
defaults to 2 and must divide `GPUS`. The model dimensions default to the released
93 layers and 896 experts/top16. That full model requires a large memory budget;
this script does not select suitable hardware or establish that it fits. Use the
normal multi-node MaxText launcher for a large distributed allocation.

For an explicitly smaller synthetic workload, after build and validation:

```bash
GPUS=4 LAYERS=4 EXPERTS=32 GRADIENT_ACCUMULATION_STEPS=2 \
  bash tests/end_to_end/gpu/kimi/run_te_exact.sh
```

The launcher uses BF16, synthetic data, sequence length 1024 by default and
`moe_qb_mask_padding=false`; configuration rejects that mask override if the
input pipeline is changed to real data. `SCAN_LAYERS=true` selects scanned
execution. `DRY_RUN=1` prints the command without launching it. The smaller
workload changes model size and is not the complete released model.

## Validation status

Prepared checks cover standalone histogram/quantile math, token masking,
microbatch additivity, empty/rejected updates, persistent bias state, native
latent MoE output parity, gradient accumulation and scanned layer statistics.
On 2026-11-10, 26 TE QB router tests passed on GPU. The six histogram/state
math tests, three native MoE/GA/scan QB tests and six GPU-initialization unit
tests passed on CPU. The `check_te_exact_moe.py` BF16 oracle passed on four
GPUs: relative RMS error was 0.67% for outputs and at most 1.17% for gradients,
within its existing 3% tolerance.

A four-layer/32-expert synthetic training workload completed with QB enabled
and without capacity overflow. Separate runs covered a small 13-layer scanned
model with gradient accumulation of two and masked QB. These proxy checks do
not establish full-model correctness. The launcher now maps CUDA-visible
devices to local ordinals and terminates sibling ranks after a failure.

Full-model HF logit parity, wider distributed coverage, checkpoint
continuation and MXFP8 training remain to be executed. The inherited Google limitations
still apply: no autoregressive KDA state cache, no Kimi pipeline/context
parallelism, and packed sequences require the configured tokamax segment path.
This is a text-backbone integration, without the vision encoder/projector or
an implementation of the original Per-Head Muon training recipe. The optional
Tokamax Mosaic KDA path is TPU-specific; a dedicated optimized GPU KDA kernel
is not supplied here.
