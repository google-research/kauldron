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

"""Testing utils."""

import functools
import jax
from kauldron.data import utils as data_utils
from kauldron.train import trainer_lib


def assert_step_specs(trainer: trainer_lib.Trainer) -> None:
  """Check the train step run correctly (fast).

  This function run a single `trainer.trainstep.step`. This use `jax.eval_shape`
  so no computation is actually executed (only shape are checked).

  This requires `trainer.train_ds.element_spec` to be available, you'll likely
  need to mock the dataset. For example if using a TFDS dataset, you can use
  `tfds.testing.mock_data`:

  ```python
  cfg = my_config.get_config()

  ...  # Eventually mutate the `cfg`

  with tfds.testing.mock_data():
    kd.testing.assert_step_specs(trainer)
  ```

  Args:
    trainer: The trainer to test.
  """
  elem_spec = trainer.train_ds.element_spec
  elem_sharding = trainer.sharding.ds

  # Skip the `init_transform`. Indeed, restoring checkpoint (partial
  # loading) will fail inside `jax.eval_shape / `jax.jit`
  init_fn = functools.partial(trainer.init_state, skip_transforms=True)
  state_spec = jax.eval_shape(init_fn)

  # The `init_fn` already compute the `model.apply`, but this will also check
  # the optimizer, auxiliaries,... including the config keys exists.
  m_batch = data_utils.mock_batch_from_elem_spec(elem_spec, elem_sharding)
  jax.eval_shape(trainer.trainstep.step, batch=m_batch, state=state_spec)
