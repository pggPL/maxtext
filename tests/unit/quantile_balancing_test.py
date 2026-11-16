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

"""Histogram QB math and step-level state contracts."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.layers import quantile_balancing as qb
from maxtext.layers.moe import MoEBiasVar
from maxtext.common.train_state_nnx import split_for_checkpoint


@pytest.mark.parametrize("chunk_size", [1, 7, 32])
def test_histogram_masks_padding_and_is_additive(chunk_size):
  scores = np.random.default_rng(5).uniform(0.1, 0.9, (19, 8)).astype(np.float32)
  bias = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
  cutoff = np.sort(scores + bias, axis=-1)[:, -3]
  mask = np.arange(19) % 3 != 0
  bounds = np.array([-1.2, 1.2], np.float32)
  bins = np.floor((cutoff[:, None] - scores - bounds[0]) * (32 / (bounds[1] - bounds[0]))).astype(int)
  bins = np.clip(bins, 0, 31)
  expected = np.zeros((8, 32), np.int32)
  for row in np.flatnonzero(mask):
    expected[np.arange(8), bins[row]] += 1
  actual = qb.histogram(jnp.array(scores), jnp.array(cutoff), bounds, jnp.array(mask), 32, chunk_size)
  np.testing.assert_array_equal(actual, expected)
  partial = sum(
      qb.histogram(jnp.array(scores[a:b]), jnp.array(cutoff[a:b]), bounds, jnp.array(mask[a:b]), 32, chunk_size)
      for a, b in [(0, 3), (3, 12), (12, 19)]
  )
  np.testing.assert_array_equal(partial, actual)


def test_quantile_centers_bias_and_preserves_empty_layers():
  counts = jnp.array([[0, 4, 0, 0], [0, 0, 4, 0]], jnp.int32)
  bias = jnp.array([0.0, 0.0], jnp.float32)
  np.testing.assert_allclose(qb.next_bias(counts, bias, 1), [-0.25, 0.25], atol=1e-7)
  stacked_counts = jnp.stack((counts, jnp.zeros_like(counts)))
  stacked_bias = jnp.array([[0.0, 0.0], [0.1, -0.1]], jnp.float32)
  actual = jax.jit(qb.next_bias, static_argnums=2)(stacked_counts, stacked_bias, 1)
  np.testing.assert_allclose(actual[0], [-0.25, 0.25], atol=1e-7)
  np.testing.assert_array_equal(actual[1], stacked_bias[1])


def test_pool_histograms_before_recovering_quantile():
  first = jnp.array([[9, 1, 0, 0], [0, 0, 1, 9]], jnp.int32)
  second = jnp.array([[0, 0, 0, 2], [2, 0, 0, 0]], jnp.int32)
  bias = jnp.zeros(2)
  pooled = qb.next_bias(first + second, bias, 1)
  incorrect = (qb.next_bias(first, bias, 1) + qb.next_bias(second, bias, 1)) / 2
  assert not np.allclose(pooled, incorrect)


def test_bias_is_persistent_and_rejected_step_does_not_change_it():
  class Gate(nnx.Module):

    def __init__(self):
      self.bias = MoEBiasVar(jnp.array([0.1, -0.1], jnp.float32))

  class Expert(nnx.Module):

    def __init__(self):
      self.gate = Gate()

  model = Expert()
  counts = jnp.array([[[0, 4, 0, 0], [0, 0, 4, 0]]], jnp.int32)
  initial = model.gate.bias.value
  qb.record(model, counts)
  histograms = qb.extract(nnx.pop(model, qb.QBHistogram))
  qb.apply_updates(model, histograms, 1, accepted=False)
  np.testing.assert_array_equal(model.gate.bias.value, initial)
  qb.apply_updates(model, histograms, 1, accepted=True)
  np.testing.assert_allclose(model.gate.bias.value.sum(), 0.0, atol=1e-7)
  qb.record(model, counts)
  state = nnx.State({"model": nnx.state(model)})
  _, persistent, ephemeral = split_for_checkpoint(state)
  assert "model" in persistent and "model" in ephemeral
  assert "qb_histogram" not in persistent.to_pure_dict()["model"]
  np.testing.assert_array_equal(persistent.to_pure_dict()["model"]["gate"]["bias"], model.gate.bias.value)
