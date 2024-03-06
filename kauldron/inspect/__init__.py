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

"""Inspect utils."""

# pylint: disable=g-importing-member

from kauldron.inspect.graphviz_utils import get_connection_graph
from kauldron.inspect.inspect import get_batch_stats
from kauldron.inspect.inspect import get_colab_model_overview
from kauldron.inspect.inspect import json_spec_like
from kauldron.inspect.inspect import lower_trainstep
from kauldron.inspect.inspect import plot_batch
from kauldron.inspect.inspect import plot_context
from kauldron.inspect.inspect import plot_sharding
from kauldron.inspect.plotting import plot_schedules
from kauldron.inspect.profile_utils import Profiler
from kauldron.inspect.trainer_visu import show_trainer_info
