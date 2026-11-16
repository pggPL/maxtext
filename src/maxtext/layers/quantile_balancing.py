# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global-step histogram Quantile Balancing for Kimi-K3."""

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


class QBHistogram(nnx.Intermediate):
  """Per-forward statistics, excluded from optimizer and checkpoint state."""


def enabled(config):
  return getattr(config, "moe_quantile_balancing", False)


def bin_bounds(bias):
  return jnp.stack((jnp.min(bias, axis=-1) - 1.0, jnp.max(bias, axis=-1) + 1.0), axis=-1)


def histogram(scores, cutoff, bounds, token_mask, num_bins, chunk_size=256):
  """Count every valid token once per expert, with bounded temporary storage."""
  scores, cutoff = jax.lax.stop_gradient((scores, cutoff))
  num_tokens, num_experts = scores.shape
  if token_mask is None:
    token_mask = jnp.ones((num_tokens,), jnp.bool_)
  chunks = (num_tokens + chunk_size - 1) // chunk_size
  padding = chunks * chunk_size - num_tokens
  scores = jnp.pad(scores, ((0, padding), (0, 0))).reshape(chunks, chunk_size, num_experts)
  cutoff = jnp.pad(cutoff.reshape(-1), (0, padding)).reshape(chunks, chunk_size)
  token_mask = jnp.pad(token_mask.reshape(-1), (0, padding)).reshape(chunks, chunk_size)
  lower, upper = bounds[0], bounds[1]

  def observe(counts, values):
    chunk_scores, chunk_cutoff, valid = values
    bins = jnp.floor((chunk_cutoff[:, None] - chunk_scores - lower) * (num_bins / (upper - lower)))
    bins = jnp.clip(bins.astype(jnp.int32), 0, num_bins - 1)
    counts = counts.at[jnp.arange(num_experts)[None, :], bins].add(valid.astype(jnp.int32)[:, None])
    return counts, None

  counts, _ = jax.lax.scan(observe, jnp.zeros((num_experts, num_bins), jnp.int32), (scores, cutoff, token_mask))
  return counts


def route_and_observe(biased_scores, scores, bias, token_mask, topk, num_bins):
  shape = scores.shape
  scores = scores.reshape(-1, shape[-1]).astype(jnp.float32)
  selected, indices = jax.lax.top_k(biased_scores.reshape(scores.shape).astype(jnp.float32), topk + 1)
  counts = histogram(scores, selected[:, -1], bin_bounds(bias), token_mask, num_bins)
  return indices[:, :topk].reshape(*shape[:-1], topk), counts[None]


def record(module, counts):
  module.sow(QBHistogram, "qb_histogram", counts, init_fn=lambda: None, reduce_fn=lambda _, value: value)


def extract(state):
  return {
      path: value
      for path, value in traverse_util.flatten_dict(state.to_pure_dict()).items()
      if path[-1] == "qb_histogram"
  }


def empty_histograms(model, config):
  result = {}
  for path, module in nnx.iter_graph(model):
    if type(module).__name__ != "RoutedMoE":
      continue
    bias = module.gate.bias.value
    mesh = module.mesh
    token_axes = ()
    if config.te_moe_block:
      token_axes = tuple(dict.fromkeys(("data", "fsdp", module._expert_parallelism_name)))
      token_axes = tuple(axis for axis in token_axes if axis in mesh.shape)
    num_shards = 1
    for axis in token_axes:
      num_shards *= mesh.shape[axis]
    leading = bias.shape[:-1]
    counts = jnp.zeros((*leading, num_shards, bias.shape[-1], config.moe_qb_num_bins), jnp.int32)
    if mesh is not None:
      spec = P(*([None] * len(leading)), token_axes or None, None, None) if token_axes else P()
      counts = jax.lax.with_sharding_constraint(counts, NamedSharding(mesh, spec))
    result[(*path, "qb_histogram")] = counts
  return result


def next_bias(counts, bias, topk):
  """Recover and center the global-batch quantile; keep bias for an empty batch."""
  bounds = bin_bounds(bias)
  cumulative = jnp.cumsum(counts, axis=-1, dtype=jnp.int32)
  total = cumulative[..., 0, -1]
  target = total.astype(jnp.float32) * (topk / bias.shape[-1])
  rank = jnp.maximum(jnp.ceil(target).astype(jnp.int32), 1)
  selected = jnp.argmax(cumulative >= rank[..., None, None], axis=-1)
  in_bin = jnp.take_along_axis(counts, selected[..., None], axis=-1)[..., 0]
  before = jnp.take_along_axis(cumulative, selected[..., None], axis=-1)[..., 0] - in_bin
  fraction = jnp.clip((target[..., None] - before) / jnp.maximum(in_bin, 1), 0.0, 1.0)
  width = (bounds[..., 1] - bounds[..., 0]) / counts.shape[-1]
  updated = bounds[..., 0, None] + (selected + fraction) * width[..., None]
  updated -= jnp.mean(updated, axis=-1, keepdims=True)
  return jnp.where(total[..., None] > 0, updated, bias)


def apply_updates(model, histograms, topk, accepted=True):
  for path, counts in histograms.items():
    module = model
    for key in path[:-1]:
      module = module[key] if isinstance(module, (dict, list, tuple)) else getattr(module, key)
    bias = module.gate.bias
    pooled = jnp.sum(counts, axis=-3, dtype=jnp.int32)
    updated = next_bias(pooled, bias.value, topk)
    bias.value = jnp.where(accepted, updated, bias.value)
