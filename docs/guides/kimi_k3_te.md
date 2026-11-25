# Kimi-K3 TE benchmark integration

This experimental branch combines Google MaxText PR #6243
(`92498e7990bde6d79150369c57a00838d2dcf43d`) with the TE/router/QB integration
from fork PR #2. It is an isolated benchmark branch; the existing PR branches
remain unchanged.

The Google model provides KDA/MLA, SiTU, latent experts, Attention Residuals and
optional layer scanning. The TE integration adds full-hidden router inputs for
latent experts, native grouped SiTU forward/backward, and global-step Quantile
Balancing. Expert checkpoint parameters remain separate `wi_0` and `wi_1`.

TE uses `activation_type="situglu"` and default soft caps 4/25. Native grouped
activation supports BF16 between separate GEMMs. Eligible MXFP8 calls can use
cuDNN fused GLU/dGLU on SM100/SM103 with Frontend >=1.33.0; other calls use the
native grouped activation. The TE revision is pinned in
`tests/end_to_end/gpu/kimi/te_revision.txt`.

QB collects padding-aware per-expert histograms during forward passes, accumulates
them across microbatches and updates centered router biases after an accepted
optimizer step. Biases remain fixed across the microbatches of that step. QB
replaces auxiliary balancing losses and sign-based bias updates.

The BF16 comparison uses a four-layer, 32-expert/top-16 proxy with full 7168/3584
hidden/latent dimensions, sequence length 1024, four GPUs and EP2. It retains the
first dense MLP, three KDA layers, one MLA layer, shared experts and Attention
Residuals. Both frameworks process 4096 tokens per optimizer step with AdamW.
The matched benchmark sets KDA core inputs to BF16; Google's default is FP32.
This precision choice needs runtime numerical validation before accepting timings.

Full 93-layer/896-expert Kimi-K3, its original optimizer recipe, vision components,
and checkpoint parity are outside the verified scope. Google PR #6243 does not
include the separate MXFP4 conversion PR. MXFP8 throughput comparisons must match
which modules are quantized; global FP8 in MCore is not equivalent to expert-only
FP8 in MaxText.

No training or profiling results have been collected for this branch yet. Static
syntax and formatting checks do not establish runtime or numerical correctness.
