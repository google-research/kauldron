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

"""Small library for creating Flatboard dashboards.

Available as standalone (`from kauldron.utils import kdash`) or directly through
`kd.kdash`.
"""

# pylint: disable=g-importing-member

from kauldron.utils.kdash.build_utils import build_and_upload
from kauldron.utils.kdash.dashboard_utils import DashboardsBase
from kauldron.utils.kdash.dashboard_utils import MetricDashboards
from kauldron.utils.kdash.dashboard_utils import MultiDashboards
from kauldron.utils.kdash.dashboard_utils import NoopDashboard
from kauldron.utils.kdash.dashboard_utils import SingleDashboard
from kauldron.utils.kdash.plot_utils import BuildContext
from kauldron.utils.kdash.plot_utils import Plot
