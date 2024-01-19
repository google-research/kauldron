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

"""Blender dataset."""

from __future__ import annotations

import dataclasses
import functools
import json
import typing

import cv2
import dataclass_array as dca
from etils import edc
from etils import epath
from etils import etree
from projects.nerf.core import colab_cache
from projects.nerf.core import structs
from projects.nerf.data import base
import numpy as np
from PIL import Image
import visu3d as v3d


class _Metadata(typing.TypedDict):
  """Metadata json specs."""

  frames: list[_Frame]
  camera_angle_x: int


class _Frame(typing.TypedDict):
  file_path: str
  transform_matrix: list[list[float]]


@edc.dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Blender(base.SceneBuilder):
  """Blender dataset."""

  name: str
  split: str
  factor: int = 1
  white_background: bool = True

  root_dir: edc.AutoCast[epath.Path] = epath.Path(
      '~/kauldron/data/nerf_synthetic'
  )

  @functools.cached_property
  def scene(self) -> structs.Scene:
    """Load images from disk."""

    # Use the Colab cache to keep the dataset in-memory during reloads
    cache = colab_cache.get_cache(self, module=v3d)
    if cache:
      return structs.Scene(
          cams=cache['cams'],
          imgs=cache['rgb'],
      )

    json_path = self.root_dir / self.name / f'transforms_{self.split}.json'
    metadata: _Metadata = json.loads(json_path.read_text())

    frames = etree.parallel_map(
        self._load_frames,
        metadata['frames'],
        is_leaf=lambda x: isinstance(x, dict),
        progress_bar=True,
    )
    frames = dca.stack(frames)

    images = frames.image
    if self.white_background:  # Invert the background
      rgb = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
      rgb = images[..., :3]

    _, h, w, _ = rgb.shape
    camera_angle_x = float(metadata['camera_angle_x'])

    world_from_cam = v3d.Transform.from_matrix(frames.world_from_cam)
    # Normalize the blender coordinates to standard open-cv conventions
    tr = v3d.Transform(
        R=[
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
    )
    world_from_cam = world_from_cam @ tr[None, ...]
    cams = v3d.Camera(
        spec=v3d.PinholeCamera.from_focal(
            resolution=(h, w),
            focal_in_px=0.5 * w / np.tan(0.5 * camera_angle_x),
        ),
        world_from_cam=world_from_cam,
    )

    # scene_boundaries = types.BoundingBox3d(
    #     min_corner=np.array((-0.8, -1.3, -0.4)),
    #     max_corner=np.array((0.8, 1.3, 1.0)),
    # )
    # rays = cameras.pixel_centers2rays(scene_boundaries=scene_boundaries)

    cache['cams'] = cams
    cache['rgb'] = rgb
    return structs.Scene(
        cams=cams,
        imgs=rgb,
    )

  def _load_frames(self, frame: _Frame) -> _LoadedFrame:
    """Load a single frame."""
    fpath = self.root_dir / self.name / (frame['file_path'] + '.png')
    with fpath.open('rb') as f:
      image = np.asarray(Image.open(f), dtype=np.float32) / 255.0

    if self.factor == 1:
      pass
    elif self.factor == 2:
      [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
      image = cv2.resize(
          image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
      )
    else:
      raise ValueError(
          f'Blender dataset only supports factor=1 or 2, {self.factor} set.'
      )
    world_from_cam = np.asarray(frame['transform_matrix'])
    return _LoadedFrame(
        world_from_cam=world_from_cam,
        image=image,
    )


class _LoadedFrame(dca.DataclassArray):
  world_from_cam: dca.typing.FloatArray['*s 4 4']
  image: dca.typing.FloatArray['*s h w c']
