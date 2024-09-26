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

"""Wrapper around `ConfigDict` to support auto-complete/type checking.

```python
from kauldron import konfig

with konfig.imports():
  from flax import linen as nn
```

Inside the `konfig.imports()` contextmanager, imports are replaced with
dummy proxy objects which return `konfig.ConfigDict` objects when called:

```python
cfg = konfig.ConfigDict()
cfg.model = nn.Dense(features=16)  # `Dense` is actually a `ConfigDict`

cfg.model.features = 32  # ConfigDict can be mutated as usual
```

Finally, objects are resolved to their actual type with:

```python
cfg = konfig.resolve(cfg)
```
"""

# pylint: disable=g-importing-member
from kauldron.konfig.configdict_base import ConfigDict
from kauldron.konfig.configdict_base import register_aliases
from kauldron.konfig.configdict_base import register_default_values
from kauldron.konfig.configdict_proxy import resolve
from kauldron.konfig.fake_import_utils import imports
from kauldron.konfig.fake_import_utils import mock_modules
from kauldron.konfig.fake_import_utils import set_lazy_imported_modules
from kauldron.konfig.flags_utils import DEFINE_config_file
from kauldron.konfig.ref_utils import ref_copy
from kauldron.konfig.ref_utils import ref_fn
from kauldron.konfig.ref_utils import WithRef
from kauldron.konfig.utils import ConfigDictLike
from kauldron.konfig.utils import placeholder
from kauldron.konfig.utils import required
# pylint: enable=g-importing-member

# Register the default `ConfigDict` overwrites
from kauldron.konfig import default_values as _default_values  # pylint: disable=g-bad-import-order
