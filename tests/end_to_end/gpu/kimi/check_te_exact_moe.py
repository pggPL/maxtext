# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BF16 latent MoE output and full-parameter/input gradient parity on 2/4 GPUs."""

import json
import math
import os
from pathlib import Path

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.layers.latent_moe import KimiLatentMoEBlock
from maxtext.utils import max_utils, maxtext_utils, maxtext_utils_nnx, sharding


def situ(gate, up):
  g, u = gate.astype(jnp.float32), up.astype(jnp.float32)
  return (4 * jnp.tanh(g / 4) * jax.nn.sigmoid(g) * (25 * jnp.tanh(u / 25))).astype(gate.dtype)


def reference(model, x, cfg):
  routed = model.routed_experts
  score = jax.nn.sigmoid(x.astype(jnp.float32) @ routed.gate.kernel[...].astype(jnp.float32))
  ids = jax.lax.top_k(score + routed.gate.bias[...], cfg.num_experts_per_tok)[1]
  selected = jnp.take_along_axis(score, ids, axis=-1)
  weight = selected / (selected.sum(-1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor
  latent = x @ model.routed_expert_down_proj.kernel[...]
  gate = jnp.einsum("bsh,ehm->bsem", latent, routed.wi_0[...])
  up = jnp.einsum("bsh,ehm->bsem", latent, routed.wi_1[...])
  experts = jnp.einsum("bsem,emh->bseh", situ(gate, up), routed.wo[...])
  probabilities = jnp.sum(jax.nn.one_hot(ids, cfg.num_experts) * weight[..., None], axis=-2)
  y = jnp.einsum("bse,bseh->bsh", probabilities.astype(x.dtype), experts)
  y32 = y.astype(jnp.float32)
  y = (y32 * jax.lax.rsqrt(jnp.mean(y32**2, axis=-1, keepdims=True) + cfg.normalization_layer_epsilon)).astype(x.dtype)
  y = y * model.routed_expert_norm.scale[...].astype(x.dtype)
  y = y @ model.routed_expert_up_proj.kernel[...]
  shared = model.shared_expert
  return y + situ(x @ shared.wi_0.kernel[...], x @ shared.wi_1.kernel[...]) @ shared.wo.kernel[...]


def main():
  root = Path(__file__).resolve().parents[4]
  processes = int(os.environ["NNODES"])
  cfg = pyconfig.initialize(
      [None, str(root / "src/maxtext/configs/base.yml")],
      model_name="kimi-k3",
      hardware="gpu",
      log_config=False,
      override_model_config=True,
      base_emb_dim=256,
      base_num_decoder_layers=2,
      full_attn_layers=[1],
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=128,
      base_mlp_dim=512,
      base_moe_mlp_dim=128,
      moe_intermediate_size=128,
      shared_intermediate_size=128,
      shared_experts=1,
      num_shared_experts=1,
      routed_expert_hidden_size=128,
      moe_expert_input_dim=128,
      moe_router_input_dim=256,
      num_experts=8,
      num_experts_per_tok=2,
      routed_scaling_factor=1.5,
      ici_expert_parallelism=2,
      ici_fsdp_parallelism=processes // 2,
      ici_data_parallelism=1,
      per_device_batch_size=1,
      max_target_length=16,
      max_prefill_predict_length=8,
      sparse_matmul=True,
      megablox=False,
      prefuse_moe_weights=False,
      te_moe_block=True,
      quantization="",
      te_gmm_quantization="te_no_quant",
      ragged_buffer_factor=0.0,
      enable_checkpointing=False,
      base_output_directory="/tmp/kimi-te-exact-parity",
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
    max_utils.maybe_bootstrap_te_moe(cfg, mesh, {"inputs": jax.ShapeDtypeStruct((processes, 16), jnp.int32)})

    def create():
      return KimiLatentMoEBlock(
          hidden_size=cfg.emb_dim,
          num_experts=cfg.num_experts,
          top_k=cfg.num_experts_per_tok,
          routed_expert_hidden_size=cfg.routed_expert_hidden_size,
          moe_intermediate_size=cfg.moe_intermediate_size,
          num_shared_experts=cfg.num_shared_experts,
          shared_intermediate_size=cfg.shared_intermediate_size,
          routed_scaling_factor=cfg.routed_scaling_factor,
          rms_norm_eps=cfg.normalization_layer_epsilon,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          config=cfg,
          mesh=mesh,
          rngs=nnx.Rngs(0),
      )

    model = maxtext_utils_nnx.create_nnx_sharded_model(nnx.eval_shape(create), create, mesh)
    assert model.routed_experts.gate.kernel.shape == (cfg.emb_dim, cfg.num_experts)
    assert not hasattr(model.routed_experts, "wi")
    model.routed_experts.gate.bias[...] = jnp.linspace(-0.05, 0.05, cfg.num_experts, dtype=jnp.float32)
    graph, params, state = nnx.split(model, nnx.Param, ...)
    spec = sharding.create_sharding(
        mesh, ("activation_batch", "activation_norm_length", None), rules=cfg.logical_axis_rules
    )
    x = jax.jit(
        lambda: jax.random.normal(jax.random.key(2), (processes, 16, cfg.emb_dim), dtype=cfg.dtype),
        out_shardings=spec,
    )()

    def measure(native):
      def loss(p, inputs):
        block = nnx.merge(graph, p, state)
        output = reference(block, inputs, cfg) if native else block(inputs)
        return jnp.mean(output.astype(jnp.float32) ** 2), output

      return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, x)

    (loss, output), (grads, dx) = measure(False)
    (expected_loss, expected), (expected_grads, expected_dx) = measure(True)
    actual_leaves, _ = jax.tree_util.tree_flatten_with_path(grads)
    expected_leaves, _ = jax.tree_util.tree_flatten_with_path(expected_grads)
    assert [p for p, _ in actual_leaves] == [p for p, _ in expected_leaves]
    pairs = [("output", output, expected), ("input_grad", dx, expected_dx)]
    pairs += [
        (jax.tree_util.keystr(path), actual, target)
        for (path, actual), (_, target) in zip(actual_leaves, expected_leaves)
    ]
    errors = {}
    for name, actual, target in pairs:
      actual, target = actual.astype(jnp.float32), target.astype(jnp.float32)
      errors[name] = float(jnp.sqrt(jnp.sum((actual - target) ** 2) / jnp.maximum(jnp.sum(target**2), 1e-20)))
      if not math.isfinite(errors[name]) or errors[name] > 0.03:
        raise AssertionError(f"{name}: relative RMS error {errors[name]} exceeds BF16 tolerance 0.03")
    if jax.process_index() == 0:
      print(json.dumps(dict(loss=float(loss), reference_loss=float(expected_loss), relative_rms_errors=errors), indent=2))
  jax.distributed.shutdown()


if __name__ == "__main__":
  main()
