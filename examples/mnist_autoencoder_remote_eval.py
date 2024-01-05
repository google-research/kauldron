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

r"""Minimal example training a simple Autoencoder on MNIST.

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/mnist_autoencoder_remote_eval.py \
  --xp.platform=jf=2x2
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

  cfg.num_train_steps = 10_000  # Longer train step to trigger multiple evals

  for ev in cfg.evals.values():  # Run each eval on a separate XM job
    ev.run = kd.evals.RunXM(
        platform="jf=2x2",
    )

  return cfg
