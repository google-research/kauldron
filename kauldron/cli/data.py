# Copyright 2026 The kauldron Authors.
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

"""Data-related CLI commands."""

from __future__ import annotations

import dataclasses
from typing import Union

from kauldron import kontext
from kauldron.cli import cmd_utils as cu
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElementSpec(cu.SubCommand):
  """Display the element spec of the training data pipeline."""

  ds_path: str = "train_ds"

  def __call__(self):
    self.print_config_origin()
    trainer = self.trainer  # trigger config resolution
    ds = kontext.get_by_path(trainer, self.ds_path)
    batch_size = getattr(ds, "batch_size", 1)
    with (
        cu.timed("Getting element spec"),
        tfds.testing.mock_data(num_examples=batch_size),
    ):
      elem_spec = ds.element_spec
    # TODO(klausg): What formatting do we want here?
    print("")
    print(f"Dataset: {self.ds_path}")
    print("batch:")
    for k, v in kontext.flatten_with_path(elem_spec).items():
      print(f"  {k}: {v.dtype}{list(v.shape)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Data(cu.CommandGroup):
  """Data commands."""

  sub_command: Union[ElementSpec]  # Union required for simple_parsing
