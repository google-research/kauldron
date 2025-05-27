# Copyright 2024 The kauldron Authors.
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

"""Summary utils."""

from __future__ import annotations

import dataclasses

import flax
import jax
from kauldron import kd
from kauldron.typing import XArray


@dataclasses.dataclass(kw_only=True, frozen=True)
class Print(kd.metrics.Metric):
  """Debug summary to print a tensor.

  Mostly useful for inspecting intermediate values in Colab.

  ```python
  cfg.train_summaries = {
      'batch': kd.contrib.summaries.Print(value='batch'),
      'preds': kd.contrib.summaries.Print(value='preds.logits'),
  )
  ```

  Attributes:
    value: Which tensor to print. All the sub-leafs will be printed (e.g.
      `values='batch'` will print `batch['image']`, `batch['label']`, etc.).
    first_n: Only print the first n scalars of each tensor.
  """

  value: kd.kontext.Key

  first_n: int = 5

  @flax.struct.dataclass
  class State(kd.metrics.State["Print"]):
    """Collects the first num_texts texts."""

    value: XArray["*b"]

    @classmethod
    def empty(cls) -> Print.State:
      return cls(value=None)

    def compute(self) -> None:
      for k, v in kd.kontext.flatten_with_path(self.value).items():
        key = self.parent.value
        if k:
          key = f"{key}/{k}"
        kd.utils.status.log(f"{key}: {v}")
      return None

    def merge(self, other: Print.State) -> Print.State:
      if self.value is None:
        return other
      else:
        return self

  def get_state(
      self,
      *,
      value: XArray["*b"],
  ) -> Print.State:
    # Eventually select a slice of the value
    def _slice(v):
      return v.flatten()[: self.first_n]

    if self.first_n:
      value = jax.tree.map(_slice, value)
    return self.State(value=value)
