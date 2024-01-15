# Copyright 2023 The kauldron Authors.
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

"""Structures."""

from __future__ import annotations

import functools

import dataclass_array as dca
from dataclass_array.typing import FloatArray
import visu3d as v3d
from visu3d.utils.lazy_imports import plotly_base


class Batch(dca.DataclassArray, v3d.Visualizable):
  """Batch."""
  ray: v3d.Ray  # ['*batch_size']
  rgb: FloatArray['*batch_size c']

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    """Construct the traces of the given object."""
    new_self = v3d.math.subsample(
        self, num_samples=self.ray.fig_config.num_samples
    )
    points = v3d.Point3d(
        p=new_self.ray.end,
        # TODO(epot): Should support various images formats (e.g. [-1, 1])
        rgb=new_self.rgb * 255.0,
    )
    return new_self.ray.make_traces() + points.make_traces()


class RayPreds(dca.DataclassArray):
  """Prediction of the model for a camera ray."""

  rgb: FloatArray['*batch_size c']
  depth: FloatArray['*batch_size 1']


class PointPreds(dca.DataclassArray):
  """Prediction of the model for a 3d point in space."""

  rgb: FloatArray['*batch_size c']
  density: FloatArray['*batch_size 1']


class Scene(dca.DataclassArray, v3d.Visualizable):
  """Scene."""

  cams: v3d.Camera  # ['*n']
  imgs: FloatArray['*n h w c']

  @functools.cached_property
  def rays(self) -> v3d.Ray:
    return self.cams.rays(normalize=False)

  @functools.cached_property
  def batch(self) -> Batch:  # -> Batch['*n h w']
    return Batch(
        ray=self.rays,
        rgb=self.imgs,
    )

  @functools.cached_property
  def flat_batch(self) -> Batch:  # -> Batch['(*n h w)']
    return self.batch.flatten()

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    """Construct the traces of the given object."""
    num_cams = 10
    points = v3d.Point3d(
        p=self.cams[:num_cams].rays(normalize=False).end,
        # TODO(epot): Should support various images formats (e.g. [-1, 1])
        rgb=self.imgs[:num_cams] * 255.0,
    )
    points = v3d.math.subsample(
        points, num_samples=points.fig_config.num_samples
    )
    return self.cams.make_traces() + points.make_traces()
