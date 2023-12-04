# Copyright 2023 The kauldron Authors.
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

"""Kauldron public API.

```python
from kauldron import kd
```
"""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top

import sys

pytest = sys.modules.get('pytest')
if pytest:
  # Inside tests, rewrite `assert` statement for better debug messages
  pytest.register_assert_rewrite('kauldron.utils.assert_utils')
del pytest, sys

# Namespaces
from kauldron import checkpoints as ckpts
from kauldron import data
from kauldron import evals
from kauldron import konfig
from kauldron import kontext
from kauldron import losses
from kauldron import metrics
from kauldron import modules as nn
from kauldron import optim
from kauldron import random
from kauldron import summaries
from kauldron import train
from kauldron import typing
from kauldron.utils import inspect
from kauldron.utils import xmanager as xm
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member

# Register aliases for cleaner config display
konfig.register_aliases(
    {
        'kauldron.kd': 'kd',
        'kauldron.kxm': 'kxm',
    }
)

# TODO(epot): This could be optional for the top-level module
# Automated documentation info
# See: https://github.com/conchylicultor/sphinx-apitree
__apitree__ = dict(
    is_package=True,
)
