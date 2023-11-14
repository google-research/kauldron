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

"""Train."""

# pylint: disable=g-importing-member
from kauldron.train.config_lib import Trainer
from kauldron.train.evaluators import Evaluator
from kauldron.train.rngs_lib import RngStream
from kauldron.train.rngs_lib import RngStreams
from kauldron.train.train_step import TrainState
from kauldron.train.train_step import TrainStep
from kauldron.utils.context import Context  # TODO(epot): Move context to train
# pylint: enable=g-importing-member


# def __getattr__(name: str):
#   import traceback

#   if name == 'Config':
#     traceback.print_stack(limit=3)
#     print('kd.train.Trainer has been renamed to kd.train.Trainer.')
#   raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
