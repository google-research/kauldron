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

"""Simple metrics for image evaluation."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax
import flax.struct
import jax
from jax import numpy as jnp
import jax.scipy as jsp
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def rescale_image(
    x: Float["*b h w c"], in_vrange: tuple[float, float]
) -> Float["*b h w c"]:
  """Rescale an image from in_vrange to (0, 1)."""
  vmin, vmax = in_vrange
  return (x - vmin) / (vmax - vmin)


@typechecked
def psnr(
    a: Float["*b h w c"],
    b: Float["*b h w c"],
    dynamic_range: float = 1.0,
) -> Float["*b 1"]:
  mse = jnp.square(a - b).mean(axis=(-3, -2, -1))
  return 20.0 * jnp.log10(dynamic_range) - 10.0 * jnp.log10(mse[..., None])


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Psnr(base.Metric):
  """PSNR."""

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None

  in_vrange: tuple[float, float] = (0.0, 1.0)
  clip: float | None = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred: Float["*b h w c"],
      target: Float["*b h w c"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Psnr.State:
    dynamic_range = self.in_vrange[1] - self.in_vrange[0]
    values = psnr(a=pred, b=target, dynamic_range=dynamic_range)
    if self.clip is not None:
      values = jnp.minimum(values, self.clip)
    return self.State.from_values(values=values, mask=mask)


# https://github.com/google-research/google-research/blob/abe03104c849ca228af386d785027809d7976a8c/jaxnerf/nerf/utils.py#L278
@typechecked
def _compute_ssim(
    img0: Float["*b h w c"],
    img1: Float["*b h w c"],
    max_val: float,
    filter_size: int,
    filter_sigma: float,
    k1: float,
    k2: float,
) -> Float["*b 1"]:
  """Computes SSIM from two images.

  This function was modeled after tf.image.ssim, and should produce comparable
  output.

  Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.

  Returns:
    Each image's mean SSIM.
  """
  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Blur in x and y (faster than the 2D convolution).
  filt_fn1 = lambda z: jsp.signal.convolve2d(z, filt[:, None], mode="valid")
  filt_fn2 = lambda z: jsp.signal.convolve2d(z, filt[None, :], mode="valid")

  # Vmap the blurs to the tensor size, and then compose them.
  num_dims = len(img0.shape)
  map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
  for d in map_axes:
    filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
    filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
  filt_fn = lambda z: filt_fn1(filt_fn2(z))

  mu0 = filt_fn(img0)
  mu1 = filt_fn(img1)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(img0**2) - mu00
  sigma11 = filt_fn(img1**2) - mu11
  sigma01 = filt_fn(img0 * img1) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  sigma00 = jnp.maximum(0.0, sigma00)
  sigma11 = jnp.maximum(0.0, sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
  )

  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
  return ssim[..., None]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Ssim(base.Metric):
  """Structural similarity (SSIM)."""

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None

  in_vrange: tuple[float, float] = (0.0, 1.0)
  filter_size: int = 11
  filter_sigma: float = 1.5
  k1: float = 0.01
  k2: float = 0.03

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred: Float["*b h w c"],
      target: Float["*b h w c"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Ssim.State:
    rescale = lambda x: rescale_image(x, self.in_vrange)
    values = _compute_ssim(
        rescale(pred),
        rescale(target),
        max_val=1.0,
        filter_size=self.filter_size,
        filter_sigma=self.filter_sigma,
        k1=self.k1,
        k2=self.k2,
    )
    return self.State.from_values(values=values, mask=mask)
