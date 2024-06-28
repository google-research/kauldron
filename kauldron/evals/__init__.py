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

from etils import epy as _epy

# pylint: disable=g-importing-member,g-import-not-at-top

# Lazy-import is import here as `run_strategies` is imported from kxm and
# we do not want to trigger a full import.
with _epy.lazy_api_imports(globals()):
  from kauldron.evals.evaluators import CollectionKeys
  from kauldron.evals.evaluators import Evaluator
  from kauldron.evals.evaluators import EvaluatorBase
  from kauldron.evals.fewshot_evaluator import FewShotEvaluator

  # Available run strategies (`Evaluator.run=`)
  from kauldron.evals.run_strategies import EveryNSteps
  from kauldron.evals.run_strategies import Once
  from kauldron.evals.run_strategies import RunStrategy
  from kauldron.evals.run_strategies import StandaloneEveryCheckpoint
  from kauldron.evals.run_strategies import StandaloneLastCheckpoint
