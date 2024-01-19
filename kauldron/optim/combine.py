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

"""Helpers for combining gradient transformations."""

from __future__ import annotations

from kauldron.typing import Schedule  # pylint: disable=g-importing-member
import optax

ScalarOrSchedule = float | int | Schedule


def named_chain(
    **transforms: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
  """Wraps optax.named_chain and allows passing transformations as kwargs.

  Example Usage:
  ```
  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adam": optax.scale_by_adam(b1=0.95),
      "decay": optax.add_decayed_weights(weight_decay=0.1),
      "lr": kd.optim.scale_by_learning_rate(0.003),
  })
  ```
  The advantages of this over using optax.chain are:
    1) Readability of the config and the sweeps because the path becomes
       "optimizer.adam.b1" rather than "optimizer[1].b1".
    2) The state of the optimizer (as stored in the checkpoint and in context)
       becomes a dictionary instead of a tuple. So it is easier to understand,
       access and manipulate.

  Args:
    **transforms: A list of GradientTransformations with names passed as kwargs.

  Returns:
    An optax.GradientTransformation that corresponds to applying the list of
    transformations in sequence.
  """
  transforms = tuple((name, transf) for name, transf in transforms.items())
  return optax.named_chain(*transforms)
