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

"""Noop handler."""

import pathlib

from kauldron.data.iterators import noop_handler
from orbax import checkpoint as ocp


def test_orbax(tmp_path: pathlib.Path):
  mgr = ocp.CheckpointManager(tmp_path)
  mgr.save(args=noop_handler.NoopArg(123), step=0)
  mgr.wait_until_finished()

  mgr = ocp.CheckpointManager(tmp_path)
  assert mgr.restore(args=noop_handler.NoopArg(567), step=0) == 567
