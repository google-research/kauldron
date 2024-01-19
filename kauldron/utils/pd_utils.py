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

"""`pandas` utils."""

from typing import Any, Optional

import pandas as pd

# Should be `pandas.io.formats.style.Styler`, but is a costly import
Styler = Any


class StyledDataFrame(pd.DataFrame):
  """`pandas.DataFrame` displayed as `pandas.io.formats.style.Styler`.

  `StyledDataFrame` is a `pandas.DataFrame` with better Jupyter notebook
  representation. Contrary to regular `pandas.DataFrame`, the `style` is
  attached to the `pandas.DataFrame`.

  ```
  df = StyledDataFrame(...)
  df.current_style.apply(...)  # Configure the style
  df  # The data-frame is displayed using ` pandas.io.formats.style.Styler`
  ```
  """

  # StyledDataFrame could be improved such as the style is forwarded when
  # selecting sub-data frames.

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)  # pytype: disable=wrong-arg-count  # re-none
    # Use name-mangling for forward-compatibility in case pandas
    # adds a `_styler` attribute in the future.
    self.__styler: Optional[Styler] = None

  @property
  def current_style(self) -> Styler:
    """Like `pandas.DataFrame.style`, but attach the style to the DataFrame."""
    if self.__styler is None:
      self.__styler = super().style  # pytype: disable=attribute-error  # re-none
    return self.__styler

  def _repr_html_(self) -> str | None:
    # See base class for doc
    if self.__styler is None:
      return super()._repr_html_()  # pytype: disable=attribute-error  # re-none
    return self.__styler._repr_html_()  # pylint: disable=protected-access
