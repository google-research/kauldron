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

r"""XManager error handling."""

import contextlib
import functools
import pdb
import traceback
from typing import TypeVar

from absl import flags
from etils import exm
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

_POST_MORTEM = flags.DEFINE_boolean(
    "catch_post_mortem",
    False,
    "Activate post-mortem debugger on error.",
)

_FnT = TypeVar("_FnT")


def catch_post_mortem(fn: _FnT) -> _FnT:
  """Decorator to report errors to XManager."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with _wu_error_handling(post_mortem=_POST_MORTEM.value):
      return fn(*args, **kwargs)

  return wrapper


@contextlib.contextmanager
def _wu_error_handling(post_mortem: bool = False):
  """Catch and log error."""
  if status.on_xmanager:
    add_artifacts = _add_xm_artifacts()
  else:
    add_artifacts = contextlib.nullcontext()

  if post_mortem:
    if status.on_xmanager:
      post_mortem_cm = g3pdb.catch_post_mortem()
    else:
      post_mortem_cm = _local_post_mortem()
  else:
    post_mortem_cm = contextlib.nullcontext()

  with post_mortem_cm:
    with add_artifacts:
      yield


@contextlib.contextmanager
def _local_post_mortem():
  """Post mortem for local runs."""
  try:
    yield
  except:  # pylint: disable=bare-except
    traceback.print_exc()
    pdb.post_mortem()


@contextlib.contextmanager
def _add_xm_artifacts():
  """Add artifacts to XManager."""
  try:
    yield
  except Exception as e:
    exc_name = type(e).__name__
    status.log_status(f"🚨 {exc_name}: {e!s}")
    status.xp.add_tags(f"🚨 {exc_name} 🚨")

    # Add links to the logs of the failing work units.
    url = exm.url_to_python_only_logs()
    wu_id = exm.current_work_unit().id
    exm.add_experiment_artifact(f"Logs failing wu={wu_id} (Python only)", url)
    raise
