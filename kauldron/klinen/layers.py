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

"""Flax layers."""

from flax import linen as nn
from kauldron.klinen import module as knn

# TODO(epot): Merge this with flax


class Dense(nn.Dense, knn.Module):  # pytype: disable=signature-mismatch
  pass


class Sequential(nn.Sequential, knn.Module):  # pytype: disable=signature-mismatch
  pass


class Dropout(nn.Dropout, knn.Module):  # pytype: disable=signature-mismatch

  @nn.compact
  def __call__(self, x):
    return super().__call__(x, deterministic=not self.training)
