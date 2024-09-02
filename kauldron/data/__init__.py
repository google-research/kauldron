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

"""Data modules.

"""

# pylint: disable=g-import-not-at-top
import etils.epy as _epy

with _epy.lazy_api_imports(globals()):
  # pylint: disable=g-importing-member,g-bad-import-order
  from kauldron.data.data_utils import IterableDataset
  # Top-level pipelines
  from kauldron.data.pipelines import Pipeline
  from kauldron.data.in_memory import InMemoryPipeline
  from kauldron.data.pipelines import PyGrainPipeline

  # PyGrain based data pipeline.
  from kauldron.data import pymix as py
  # TODO(epot): Migrate all existing symbols to `kd.data.tf.`
  # tf.data base data pipeline.
  from kauldron.data import kmix as tf

# TODO(klausg): Temporary removal until importing works
