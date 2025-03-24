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

"""Text summaries."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import flax
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import XArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


# TODO(epot): Supports text metrics in the trainer.
@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowTexts(metrics.Metric):
  """Show texts.

  Note: This metric is not jit-compatible, as it manipulates text. So it
  cannot be used in the trainer (yet). Instead, it can only be used in custom
  `kd.evals.Evaluator`, like `gm.evals.SamplerEvaluator`.

  Attributes:
    texts: Key to the text to display.
    num_texts: Number of texts to collect and display. Default 5.
  """

  texts: kontext.Key
  num_texts: int = 5

  @flax.struct.dataclass
  class State(metrics.AutoState["ShowTexts"]):
    """Collects the first num_texts texts."""

    texts: list[str] = metrics.truncate_field(num_field="parent.num_texts")

    @typechecked
    def compute(self) -> XArray["num_texts"]:
      texts = super().compute().texts
      return texts

  @typechecked
  def get_state(
      self,
      texts: str | Sequence[str] | XArray["*b"],
  ) -> ShowTexts.State:
    # TODO(epot): Ensure that the input are actually texts (and not int arrays).

    # Numpy has poor support for string arrays, so uses `dtype=object`
    texts = np.asarray(texts, dtype=object).flatten()
    return self.State(texts=texts)
