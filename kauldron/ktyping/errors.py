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

"""Dump of errors raised by ktyping."""


class UnknownDimensionError(KeyError):
  """Raised when a dimension is unknown in the current scope."""


class AmbiguousDimensionError(ValueError):  # Or KeyError?
  """Raised when a dimension is ambiguous within the current scope."""


class DimLengthError(ValueError):
  """Raised when a dimension has the wrong length."""  # single dim / tuple


class IncompatibleDimensionError(ValueError):
  """Raised when a dimension has an incompatible value in the current scope."""


class NoActiveScopeError(RuntimeError):  # Or AssertionError?
  """Raised when there is no active scope."""
