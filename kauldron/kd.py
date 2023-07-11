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

# pylint: disable=unused-import,g-importing-member

# Namespaces
from kauldron import data
from kauldron import konfig
from kauldron import losses
from kauldron import metrics
from kauldron import modules as nn
from kauldron import random
from kauldron import summaries
from kauldron import train
from kauldron import typing
from kauldron.core import get_by_path
from kauldron.core import tree_flatten_with_path
from kauldron.utils import plotting
from kauldron.utils import xmanager as xm

# Register aliases for cleaner config display
konfig.register_aliases(
    {
        'kauldron.kd': 'kd',
    }
)
