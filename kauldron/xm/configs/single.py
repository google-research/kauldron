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

r"""kxm config to launch a single job.

Usage:

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
 --xp=third_party/py/kauldron/xm/configs/single.py \
 --xp.use_interpreter \
 --xp.target=//path/to/my:target
```

See for all the available flags.
"""

from kauldron import konfig

with konfig.imports():
  from kauldron import kxm  # pylint: disable=g-import-not-at-top


def get_config():
  # Job parameters can be set both in `jobs` or directly at
  # the top-level.
  # See for all the o
  return kxm.Experiment(
      jobs={
          'main': kxm.Job(),
      },
  )
