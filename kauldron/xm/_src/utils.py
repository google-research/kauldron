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

"""Utils."""

from collections.abc import Iterator
import contextlib
from typing import Any, Callable, TypeVar

from etils import epy

_FnT = TypeVar('_FnT')


# TODO(epot): Add to `epy.reraise_fn`
def reraise_fn(msg: str, *fargs: Any, **fkwargs: Any) -> Callable[[_FnT], _FnT]:
  """Wrap a function in a `epy.reraise` for better debug messages."""

  def decorator(fn):
    def decorated(*args, **kwargs):
      try:
        return fn(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught
        epy.reraise(e, prefix=msg.format(*fargs, **fkwargs))

    return decorated

  return decorator


@contextlib.contextmanager
def maybe_log_colab() -> Iterator[None]:
  """Display log in a collapsible section in Colab."""
  if not epy.is_notebook():
    yield
  else:
    # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    from etils import ecolab  # pylint: disable=g-import-not-at-top
    from colabtools import googlelog  # pylint: disable=g-import-not-at-top
    # pylint: enable=g-import-not-at-top  # pytype: enable=import-error

    with ecolab.collapse('Build output'):
      with googlelog.Capture():
        yield
