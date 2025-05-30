# Copyright 2025 The kauldron Authors.
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

"""Json utils."""

import json
from typing import Any

from etils import epy
from kauldron import konfig


def arg_to_json(v: Any) -> str:
  """Converts an object to a JSON string."""
  # Note that `str` are passed as-is.
  # If the string is itself JSON, it will be decoded by `arg_from_json` which
  # could be an issue.
  # This is because there's no way to infer that `--xx=123` should be
  # decoded as `int` or `str`.
  return v if isinstance(v, str) else _JsonEncoder().encode(v)


def arg_from_json(v: str) -> epy.typing.Json:
  """Decodes the JSON string or returns the string itself."""
  # The decoded values should always have been encoded JSON strings from
  # `_serialize_job_kwargs`, so there shouldn't be risk of badly formatted JSON.
  try:
    return json.loads(v)
  except json.JSONDecodeError:
    return v


class _JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, konfig.ConfigDict):
      return json.loads(o.to_json())
    else:
      return super().default(o)
