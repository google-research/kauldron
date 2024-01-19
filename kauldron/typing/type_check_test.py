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

from kauldron.typing import Float, TypeCheckError, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pytest


def test_decorator():
  @typechecked
  def _foo(_: Float["a"]) -> Float["a"]:
    return np.zeros((1,))

  _foo(np.zeros((1,)))

  with pytest.raises(TypeCheckError):
    _foo(np.zeros((1, 2)))

  with pytest.raises(TypeCheckError):
    _foo(np.zeros((8,)))


def test_decorator2():
  @typechecked
  def _foo(x: Float["a 1"], y: Float["a b"]) -> Float["a/2 b+1"]:  # pylint: disable=unused-argument
    return np.zeros((5, 4))

  _foo(np.zeros((10, 1)), np.zeros((10, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((10, 1)), y=np.zeros((8, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((6, 1)), y=np.zeros((6, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((10, 2)), y=np.zeros((10, 3)))
