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

"""Kauldron XManager launcher public API."""

# pylint: disable=g-importing-member,unused-import,g-bad-import-order

# Root experiment object
from kauldron.xm._src.experiment import Experiment

# Jobs builder
from kauldron.xm._src.jobs_info import JobsInfo
from kauldron.xm._src.jobs_info import Jobs
from kauldron.xm._src.kauldron_utils import KauldronJobs

# Single job info
from kauldron.xm._src.job_lib import Debug
from kauldron.xm._src.job_lib import Job
from kauldron.xm._src.requirements import JobRequirements

# Sweep info
from kauldron.xm._src.sweep_utils import SweepInfo
from kauldron.xm._src.sweep_utils import NoSweep
from kauldron.xm._src.sweep_utils import SimpleSweep
from kauldron.xm._src.kauldron_utils import KauldronSweep
from kauldron.xm._src.sweep_utils import SweepItem

# Others
from kauldron.xm._src.dir_utils import SubdirFormat
from kauldron.xm._src.dir_utils import WU_DIR_PROXY
from kauldron.xm._src.dir_utils import XP_DIR_PROXY
from kauldron.xm._src.dir_utils import file_path

# Register aliases for cleaner config display
from kauldron import konfig

konfig.register_aliases({
    'kauldron.kxm': 'kxm',
})
del konfig
