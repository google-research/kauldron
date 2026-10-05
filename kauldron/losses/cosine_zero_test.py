# Copyright 2026 The kauldron Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finite derivatives for epsilon-regularized cosine loss at zero vectors."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from kauldron.losses import simple
import numpy as np
import optax


class CosineZeroTest(parameterized.TestCase):

  @parameterized.parameters(1e-8, 1e-3, 0.1)
  def test_zero_prediction_has_the_analytic_finite_gradient(self, eps):
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t', eps=eps)
    target = jnp.array([[3.0, 4.0]])
    expected = -np.asarray(target) / ((5.0 + eps) * eps)
    objective = lambda p: loss.get_values(p, target).sum()
    for value_grad in (
        jax.value_and_grad(objective),
        jax.jit(jax.value_and_grad(objective)),
    ):
      value, grad = value_grad(jnp.zeros_like(target))
      self.assertEqual(float(value), 0.0)
      np.testing.assert_allclose(grad, expected, rtol=2e-6, atol=0)

  @parameterized.parameters(1e-8, 0.1)
  def test_zero_target_has_the_analytic_finite_gradient(self, eps):
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t', eps=eps)
    pred = jnp.array([[3.0, 4.0]])
    grad = jax.jit(jax.grad(lambda t: loss.get_values(pred, t).sum()))(
        jnp.zeros_like(pred)
    )
    np.testing.assert_allclose(
        grad, -np.asarray(pred) / ((5.0 + eps) * eps), rtol=2e-6
    )

  def test_two_zero_vectors_have_zero_value_and_finite_zero_gradients(self):
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t')
    objective = lambda p, t: loss.get_values(p, t).sum()
    p = jnp.zeros((2, 3))
    value, gradients = jax.value_and_grad(objective, argnums=(0, 1))(p, p)
    self.assertEqual(float(value), 0.0)
    for gradient in gradients:
      np.testing.assert_array_equal(gradient, np.zeros_like(p))

  def test_masked_zero_embeddings_do_not_poison_parameter_gradients(self):
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t', eps=0.1)
    p = jnp.array([[0.0, 0.0], [2.0, 1.0]])
    t = jnp.array([[1.0, 0.0], [1.0, 2.0]])
    mask = jnp.array([[0.0], [1.0]])
    objective = lambda x: loss.get_state(
        preds=x, targets=t, mask=mask
    ).compute()
    value, grad = jax.jit(jax.value_and_grad(objective))(p)
    self.assertTrue(bool(jnp.isfinite(value)))
    self.assertTrue(bool(jnp.isfinite(grad).all()))
    np.testing.assert_array_equal(grad[0], np.zeros(2))
    reference = jax.grad(lambda x: loss.get_values(x, t[1:]).sum())(p[1:])
    np.testing.assert_allclose(grad[1:], reference)

  def test_sgd_can_leave_a_zero_initial_embedding(self):
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t', eps=0.1)
    p = jnp.zeros((1, 2))
    t = jnp.array([[3.0, 4.0]])
    objective = lambda x: loss.get_state(preds=x, targets=t).compute()
    optimizer = optax.sgd(0.01)
    value, grad = jax.value_and_grad(objective)(p)
    update, _ = optimizer.update(grad, optimizer.init(p))
    new_p = optax.apply_updates(p, update)
    self.assertTrue(bool(jnp.isfinite(new_p).all()))
    self.assertLess(float(objective(new_p)), float(value))

  @parameterized.parameters(
      {'shape': (2, 3)}, {'shape': (2, 1, 3)}, {'shape': (3,)}
  )
  def test_nonzero_values_and_gradients_match_original_formula(self, shape):
    p = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1
    t = jnp.flip(p, axis=-1) + 0.5
    loss = simple.NegativeCosineSimilarity(preds='p', targets='t', eps=0.1)

    def original(x):
      a = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 0.1)
      b = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 0.1)
      return -jnp.sum(a * b)

    actual = jax.value_and_grad(lambda x: loss.get_values(x, t).sum())(p)
    expected = jax.value_and_grad(original)(p)
    for a, b in zip(actual, expected):
      np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
