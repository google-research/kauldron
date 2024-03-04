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

r"""Minimal example of model parallelism (API might change in the future).

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/mnist_autoencoder_sharded.py \
  --xp.use_interpreter \
  --xp.platform=df=2x2
```

"""

from kauldron import konfig
from examples import mnist_autoencoder

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = mnist_autoencoder.get_config()

  # TODO(epot): Use more complicated sharding.
  cfg.sharding = kd.sharding.ShardingStrategy(
      params=kd.sharding.REPLICATED,
  )

  return cfg
