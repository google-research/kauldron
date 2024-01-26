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

import sys
from unittest import mock

import apitree


# TODO(epot): Delete once `grain` can be imported
sys.modules['grain'] = mock.MagicMock()
sys.modules['grain._src'] = mock.MagicMock()
sys.modules['grain._src.core'] = mock.MagicMock()
sys.modules['grain._src.core.constants'] = mock.MagicMock()
sys.modules['grain._src.tensorflow'] = mock.MagicMock()
sys.modules['grain._src.tensorflow.transforms'] = mock.MagicMock()
sys.modules['grain.tensorflow'] = mock.MagicMock()

import grain.tensorflow as _mocked_grain  # pylint: disable=g-import-not-at-top


class _MockedTransform:
  pass


# Required for inheritance `class MyTransform(grain.MapTransform)`
_mocked_grain.MapTransform = _MockedTransform
_mocked_grain.RandomMapTransform = _MockedTransform


# Early failure if kauldron cannot be imported
# Read-the-doc install kauldron not in `-e` edit mode, so should only import
# kauldron after `apitree` import kauldron from the right path.
# from kauldron import kd  # pylint: disable=g-import-not-at-top


apitree.make_project(
    modules=apitree.ModuleInfo(
        api='kauldron.kd',
        module_name='kauldron',  # What to include
        alias='kd',
    ),
    includes_paths={
        'kauldron/konfig/docs/demo.ipynb': 'konfig.ipynb',
        'kauldron/klinen/README.md': 'klinen.md',
    },
    globals=globals(),
)
