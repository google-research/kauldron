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

"""Basic top level modules."""

from __future__ import annotations

import collections.abc
import copy
from typing import Any, Callable, Optional, Sequence

import einops
from flax import linen as nn
from kauldron import kontext
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class Sequential(nn.Module):
  """Like nn.Sequential but allows configuring input and output keys.

  Attributes:
    inputs: The input key used to pass into the model. E.g. "batch.image".
    outputs: Optional key to wrap the output in. If set, then the return value
      is a dictionary {outputs: value}. Otherwise just the value is returned.
    layers: A sequence of callables (usually `nn.Modules`) to execute.
  """

  layers: Sequence[Callable[..., Any]] = ()
  inputs: kontext.Key = kontext.REQUIRED  # required only if top-level module
  outputs: Optional[str] = None

  @nn.compact
  def __call__(self, inputs):
    outputs = inputs
    for layer in self.layers:
      if isinstance(outputs, tuple):
        outputs = layer(*outputs)
      elif isinstance(outputs, collections.abc.Mapping):
        outputs = layer(**outputs)
      else:
        outputs = layer(outputs)

    if self.outputs is None:
      return outputs
    else:
      return {self.outputs: outputs}

  @classmethod
  def from_repeated(
      cls,
      layer: nn.Module,
      repeats: int,
      *,
      shared_weights: bool = False,
      inputs: kontext.Key = kontext.REQUIRED,
      outputs: Optional[str] = None,
      prefix: Sequence[nn.Module] = (),
      suffix: Sequence[nn.Module] = (),
  ) -> Sequential:
    """Create a sequential Module by repeating a given layer module.

    Effectively constructs a Sequential module with:
    layers = prefix + [layer] * repeats + suffix

    Args:
      layer: The nn.Module to be repeated
      repeats: How often to repeat the given layer
      shared_weights: Wether the repeated layers should share weigths.
      inputs: The input key (see module doc).
      outputs: Optional key to wrap the output in (see module doc).
      prefix: A sequence of modules to be added before the repeated layer.
      suffix: A sequence of modules to be added after the repeated layer.

    Returns:
      A Sequential Module.
    """
    repeated_layer = [
        layer if shared_weights else copy.deepcopy(layer)
        for _ in range(repeats)
    ]
    layers = tuple(prefix) + tuple(repeated_layer) + tuple(suffix)
    return Sequential(inputs=inputs, outputs=outputs, layers=layers)


class FlatAutoencoder(nn.Module):
  """Very simple auto-encoder class to showcase using keys and submodules."""

  encoder: nn.Module
  decoder: nn.Module

  # If used as top-module this key will be used to fill the argument of __call__
  inputs: kontext.Key = kontext.REQUIRED  # e.g. 'batch.image'

  @typechecked
  @nn.compact
  def __call__(self, inputs: Float['b *inner']) -> dict[str, Float['b *inner']]:
    flat_inputs = einops.rearrange(inputs, 'b ... -> b (...)')

    h = self.encoder(flat_inputs)
    y = self.decoder(h)

    y = y.reshape(inputs.shape)

    # Use the same key as the inputs for the outputs (but without batch prefix)
    # TODO(klausg): add an OutKey or similar to make this configurable
    return {self.inputs.removeprefix('batch.'): y}
