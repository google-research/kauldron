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

"""Shape-spec related utilities."""

import dataclasses
import jaxtyping


class ShapeError(ValueError):
  pass


@dataclasses.dataclass
class Memo:
  """Jaxtyping information about the shapes in the current scope."""

  single: dict[str, int]
  variadic: dict[str, tuple[int, ...]]

  @classmethod
  def from_current_context(cls):
    """Create a Memo from the current typechecking context."""
    single_memo, variadic_memo, *_ = jaxtyping._storage.get_shape_memo()  # pylint: disable=protected-access

    variadic_memo = {k: tuple(dims) for k, (_, dims) in variadic_memo.items()}
    return cls(
        single=single_memo.copy(),
        variadic=variadic_memo.copy(),
    )

  def __repr__(self) -> str:
    out = {k: v for k, v in self.single.items()}
    out.update({f'*{k}': v for k, v in self.variadic.items()})
    return repr(out)
