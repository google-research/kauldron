# Copyright 2025 The kauldron Authors.
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

"""Test."""

from jax import numpy as jnp
from kauldron import metrics
import numpy as np
import pytest


def test_roc():
  metric = metrics.RocAuc()

  logits = jnp.asarray([
      [1.0, 0.0, 0.0],
      [0.3, 0.4, 0.3],
      [0.0, 2.0, 8.0],
  ])
  labels = jnp.asarray([[1], [0], [2]], dtype=jnp.int32)

  s0 = metric.get_state(logits=logits, labels=labels)
  s1 = metrics.RocAuc().empty()

  x = s0.merge(s1).finalize().compute()
  y = s1.merge(s0).finalize().compute()
  np.testing.assert_allclose(x, y)

  # Parent is correctly forwarded
  assert s0.parent == s1.parent
  assert s0.merge(s1).parent == s0.parent

  metric = metrics.RocAuc(multi_class_mode='ovo')
  s3 = metric.get_state(logits=logits, labels=labels)
  with pytest.raises(ValueError, match='from different metrics'):
    s3.merge(s0)
  with pytest.raises(ValueError, match='from different metrics'):
    s0.merge(s3)
