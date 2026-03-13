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

import pickle
import dill
from kauldron.konfig import utils
import pytest


@pytest.mark.parametrize('pickler', [pickle, dill])
def test_frame_stack_is_picklable(pickler):
  stack = utils.FrameStack.from_current()
  pickled = pickler.dumps(stack)
  unpickled = pickler.loads(pickled)
  # Unpickling always produces an empty instance.
  assert isinstance(unpickled, utils.FrameStack)
  assert len(unpickled) == 0
