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

"""Requirement wrapper."""

import functools

from xmanager import xm


def platform_from_requirements(requirements: xm.JobRequirements) -> str:
  """Extract the `platform` string (e.g. `jf=2x2`) from the requirements."""
  if requirements.accelerator is None:
    return 'cpu'
  accelerator = _resource_type_to_lower()[requirements.accelerator]
  assert requirements.topology is not None
  topology = requirements.topology.name
  return f'{accelerator}={topology}'


@functools.cache
def _resource_type_to_lower() -> dict[xm.ResourceType, str]:
  """Lower alias associated with each type (e.g. `JELLYFISH` -> `jf`)."""
  # e.g. `_resource_type_to_lower()[xm.ResourceType.JELLYFISH] == 'jf'`
  resource_type_to_short_name = {}
  for k, v in xm.ResourceType.__dict__.items():
    if not isinstance(v, xm.ResourceType):
      continue
    if v not in resource_type_to_short_name:
      resource_type_to_short_name[v] = k
    else:
      # Update the name if a shorted is found (e.g. `JELLYFISH` vs `JF`)
      curr_name = resource_type_to_short_name[v]
      if len(k) < len(curr_name):
        curr_name = k
      resource_type_to_short_name[v] = curr_name

  resource_type_to_short_name = {
      v: k.lower() for v, k in resource_type_to_short_name.items()
  }
  return resource_type_to_short_name
