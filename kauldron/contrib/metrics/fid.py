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

"""FID Metrics that takes as input a statistics dict for the train set."""

from __future__ import annotations

import dataclasses
from functools import cached_property  # pylint: disable=g-importing-member
from typing import Callable, Optional  # pylint: disable=g-multiple-imports

import einops
from etils import epath
import flax
import jax
from kauldron import kontext
from kauldron import metrics
from kauldron.metrics import base
from kauldron.metrics import image as image_metrics
from kauldron.metrics.fid import _get_fid_score  # pylint: disable=g-importing-member
from kauldron.metrics.fid import _get_stats_for_fid  # pylint: disable=g-importing-member
from kauldron.metrics.fid import _load_cached_params  # pylint: disable=g-importing-member
from kauldron.metrics.fid import _run_model  # pylint: disable=g-importing-member
from kauldron.typing import Bool  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FidWithStats(base.Metric):
  """Fréchet Inception Distance (FID), see https://arxiv.org/abs/1706.08500."""

  pred: kontext.Key = kontext.REQUIRED
  reference_stats_loader: Callable[[], dict[str, jax.Array]]
  mask: Optional[kontext.Key] = None
  in_vrange: tuple[float, float] = (0.0, 1.0)
  cns_dump_file: Optional[str] = None

  @cached_property
  def reference_stats(self) -> dict[str, jax.Array]:
    return self.reference_stats_loader()

  @flax.struct.dataclass
  class State(metrics.AutoState["FidWithStats"]):
    """FID state."""

    pred_feats: Float["b h w d"] = metrics.concat_field()

    @typechecked
    def compute(self) -> float:
      pred_feats = self.pred_feats
      if self.parent.cns_dump_file is not None:
        fpath = epath.Path(self.parent.cns_dump_file)
        fpath.parent.mkdir(exist_ok=True, parents=True)
        with fpath.open("wb") as f:
          np.save(f, pred_feats)
      pred_mu, pred_sigma = _get_stats_for_fid(pred_feats, mean_only=False)
      target_mu = self.parent.reference_stats["mean"]
      target_sigma = self.parent.reference_stats["cov"]

      return _get_fid_score(pred_mu, pred_sigma, target_mu, target_sigma)

  @typechecked
  def get_state(
      self,
      pred: Float["*b h w c"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> FidWithStats.State:
    if mask is not None:
      raise ValueError("Mask is currently not supported for FID.")

    params = _load_cached_params()

    flatten = lambda x: einops.rearrange(x, "... h w c -> (...) h w c")
    rescale = lambda x: image_metrics.rescale_image(x, self.in_vrange) * 255.0

    pred = flatten(rescale(pred))
    pred_feats = _run_model(params, pred)

    return self.State(pred_feats=pred_feats)
