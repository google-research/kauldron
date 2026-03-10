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

"""Testing the behavior of the KdNnxModule."""

import dataclasses
from functools import partial  # pylint: disable=g-importing-member
from flax import linen
from flax import nnx
import jax.numpy as jnp
from kauldron import kd
from kauldron.contrib.modules import knnx


def test_knnx_module():
  # first, create simple linen module
  class A(linen.Module):
    input_dim: int = 10
    hdim: int = 10

    @linen.compact
    def __call__(self, x):
      return linen.Dense(self.hdim)(x)

  # now, create nnx module
  @dataclasses.dataclass(kw_only=True)
  class B(knnx.KdNnxModule):
    input_dim: int = 10
    hdim: int = 10

    def setup(self, rngs: nnx.Rngs = nnx.Rngs(0)):
      self.lin = nnx.Linear(self.input_dim, self.hdim, rngs=rngs)

    def __call__(self, x):
      return self.lin(x)

  # now, create the modules
  linen_mod = A(input_dim=10, hdim=10)
  kd_nnx_mod = B(input_dim=10, hdim=10)

  # check the interface is the same
  linen_vars = linen_mod.init(
      {'params': kd.random.PRNGKey(0)}, jnp.ones((1, 10))
  )
  kd_nnx_vars = kd_nnx_mod.init(
      {'params': kd.random.PRNGKey(0)}, jnp.ones((1, 10))
  )
  output_linen, _ = linen_mod.apply(
      linen_vars, jnp.ones((1, 10)), mutable=True, rngs={}
  )
  output_kd_nnx, _ = kd_nnx_mod.apply(
      kd_nnx_vars, jnp.ones((1, 10)), mutable=True, rngs={}
  )
  assert 'params' in kd_nnx_vars, 'params not found in kd_nnx_vars.'
  assert 'nnx' in kd_nnx_vars, 'nnx not found in kd_nnx_vars.'
  assert (
      'intermediates' in kd_nnx_vars
  ), 'intermediates not found in kd_nnx_vars.'
  assert (
      output_linen.shape == output_kd_nnx.shape
  ), 'NNX wrapper output shape is different from linen module.'


def test_knnx_module_from_def():
  """Test existing nnx module with LinenModuleFromNnxDef."""

  class MyNnxModule(nnx.Module):

    def __init__(self, input_dim, hdim, rngs: nnx.Rngs = nnx.Rngs(0)):
      super().__init__()
      self.lin = nnx.Linear(input_dim, hdim, rngs=rngs)

    def __call__(self, x):
      return self.lin(x)

  kd_mod = knnx.LinenModuleFromNnxDef(
      nnx_init=partial(MyNnxModule, input_dim=3, hdim=3),
  )
  variables = kd_mod.init({'params': kd.random.PRNGKey(0)}, jnp.ones((1, 3)))
  out, _ = kd_mod.apply(variables, jnp.ones((1, 3)), mutable=True)
  assert out.shape == (1, 3)


def test_capture_intermediates():

  @dataclasses.dataclass(kw_only=True)
  class ModuleWithIntermediates(knnx.KdNnxModule):
    input_dim: int = 3
    hdim: int = 5

    def setup(self, rngs: nnx.Rngs = nnx.Rngs(0)):
      self.lin = nnx.Linear(self.input_dim, self.hdim, rngs=rngs)

    def __call__(self, x):
      h = self.lin(x)
      self.sow(nnx.Intermediate, 'hidden', h)
      return h

  mod = ModuleWithIntermediates(input_dim=3, hdim=5)
  variables = mod.init({'params': kd.random.PRNGKey(0)}, jnp.ones((1, 3)))
  out, variables = mod.apply(
      variables,
      jnp.ones((1, 3)),
      mutable=True,
      capture_intermediates=True,
  )
  assert out.shape == (1, 5)
  assert 'intermediates' in variables
  assert variables['intermediates']


def test_rng_determinism():

  @dataclasses.dataclass(kw_only=True)
  class ModuleWithDropout(knnx.KdNnxModule):
    input_dim: int = 3
    hdim: int = 1000
    output_dim: int = 3

    def setup(self, rngs: nnx.Rngs = nnx.Rngs(0)):
      self.lin = nnx.Linear(self.input_dim, self.hdim, rngs=rngs)
      self.proj = nnx.Linear(self.hdim, self.output_dim, rngs=rngs)
      self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

    def __call__(self, x):
      x = self.lin(x)
      x = self.dropout(x)
      x = nnx.relu(x)
      x = self.proj(x)
      return x

  mod = ModuleWithDropout(input_dim=3, hdim=1000, output_dim=3)
  rngs = {'params': kd.random.PRNGKey(0), 'dropout': kd.random.PRNGKey(42)}
  variables = mod.init(rngs, jnp.ones((1, 3)))

  x = jnp.ones((1, 3))
  out_a, _ = mod.apply(
      variables, x, mutable=True, rngs={'dropout': kd.random.PRNGKey(1)}
  )
  out_b, _ = mod.apply(
      variables, x, mutable=True, rngs={'dropout': kd.random.PRNGKey(1)}
  )
  assert jnp.allclose(out_a, out_b), 'Same RNG should yield the same output.'

  out_c, _ = mod.apply(
      variables, x, mutable=True, rngs={'dropout': kd.random.PRNGKey(99)}
  )
  assert not jnp.allclose(
      out_a, out_c
  ), 'Different RNG should yield different output.'
