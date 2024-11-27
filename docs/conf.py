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

"""Generate documentation.

Usage (from the root directory):

```sh
pip install -e .[docs]

sphinx-build -b html docs/ docs/_build
```
"""

import apitree


apitree.make_project(
    modules=apitree.ModuleInfo(
        api='kauldron.kd',
        module_name='kauldron',  # What to include
        alias='kd',
    ),
    includes_paths={
        'kauldron/konfig/docs/demo.ipynb': 'konfig.ipynb',
        'kauldron/kontext/README.md': 'kontext.md',
        'kauldron/data/py/README.md': 'data_py.md',
        'kauldron/klinen/README.md': 'klinen.md',
        'kauldron/random/README.md': 'random.md',
    },
    globals=globals(),
)
