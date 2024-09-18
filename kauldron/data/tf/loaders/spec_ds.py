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

"""Element spec dataset."""

import dataclasses
from typing import ClassVar

from etils import enp
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax
from kauldron import random
from kauldron.data.tf import base
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
import tensorflow as tf

# Currently uses `TFDataPipeline` to support transformations but might not be
# a good idea. Maybe should have a non-TF version of this class.


@dataclasses.dataclass(frozen=True)
class ElementSpecDataset(base.TFDataPipeline):
  """Dataset which returns a single batch matching element_spec.

  Allows model initialization without having to load an actual dataset.

  Usage:

  ```python
  cfg.train_ds = kd.data.ElementSpecDataset(
      spec=kd.from_xid.get_element_spec(xid=cfg.ref.aux.xid)
  )
  ```
  """

  spec: PyTree[enp.ArraySpec]

  _supports_symbolic_checkpoint: ClassVar[bool] = False

  def __post_init__(self):
    super().__post_init__()
    if self.batch_size is not None:
      raise ValueError(
          "ElementSpecDataset should not specify batch_size but rather include"
          " it in the spec."
      )

  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    element_spec = etree.spec_like(self.spec)
    del rng  # Unused
    batch = jax.tree.map(
        lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype),
        element_spec,
    )
    return tf.data.Dataset.from_tensors(batch)
