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

"""Deprecated summaries."""


class Summary:
  """Deprecated Summaries. Raises an error if instantiated."""

  def __new__(cls, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        f"{cls.__name__}: old-style summaries are deprecated. Please migrate to"
        " the new kd.metrics.Metric based summaries. See"
        " kd.summaries.images.ShowImages for an example."
    )


class ImageSummary(Summary):
  """Deprecated ImageSummary. Raises an error if instantiated."""
