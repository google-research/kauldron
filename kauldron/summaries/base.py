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

"""Interface for custom summaries."""

from __future__ import annotations

import abc
from typing import Any

from kauldron import kontext
from kauldron.typing import Float, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

Images = Float["*b h w c"] | UInt8["*b h w c"]


class Summary(abc.ABC):
  """Base class for defining non-scalar tensorboard summaries (e.g. images)."""

  def gather_kwargs(self, context: Any) -> dict[str, Any]:
    """Returns the required information from context as a kwargs dict."""
    return kontext.resolve_from_keyed_obj(context, self)


class ImageSummary(Summary, abc.ABC):
  """Base class for image summaries."""

  def __init_subclass__(cls, **kwargs):
    msg = (
        f"{cls.__name__} will stop working by the end of 2024. "
        "Migrate to the new kd.metrics.Metric based summaries. "
        "See kd.summaries.images.ShowImages for an example."
    )
    status.warn(msg, DeprecationWarning, stacklevel=2)
    super().__init_subclass__(**kwargs)

  @abc.abstractmethod
  def get_images(self, **kwargs) -> Images:
    ...

  @typechecked
  def __call__(self, *, context: Any = None, **kwargs) -> Images:
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or keyword arguments,"
            f"but got context and {kwargs.keys()}."
        )
      kwargs = kontext.resolve_from_keyed_obj(
          context, self, func=self.get_images
      )
    return self.get_images(**kwargs)
