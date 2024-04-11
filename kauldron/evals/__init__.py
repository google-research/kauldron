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

"""Evaluator."""

# pylint: disable=g-importing-member,g-import-not-at-top

from kauldron.evals.evaluators import Evaluator
from kauldron.evals.evaluators import EvaluatorBase
from kauldron.evals.fewshot_evaluator import FewShotEvaluator

# RunStrategy are available in both XM and Kauldron side
from kauldron.xm._src.run_strategies import RunEvery
from kauldron.xm._src.run_strategies import RunOnce
from kauldron.xm._src.run_strategies import RunSharedXM
from kauldron.xm._src.run_strategies import RunStrategy
from kauldron.xm._src.run_strategies import RunXM
