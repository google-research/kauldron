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

r"""Base config for Kauldron codebase.

This can serve as an example of base-config for other non-Kauldron projects.

Usage:

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
 --cfg=third_party/py/projects/my_project/config/my_config.py
```

"""

from kauldron import konfig

with konfig.imports():
  # pylint: disable=g-import-not-at-top
  from kauldron import kxm
  from xmanager import xm_abc
  # pylint: enable=g-import-not-at-top


def get_config():
  return kxm.Experiment(
      root_dir='/path/to/home/{author}/kd/',
      # Job builder is the Job provider
      jobs_provider=kxm.KauldronJobs(),
      # `KauldronSweep` is activated either:
      # * By setting `--xp.sweep` (for `def sweep()`)
      # * By setting `--xp.sweep=aaa,bbb` (named sweep `def sweep_aaa()`,...)
      sweep_info=kxm.KauldronSweep(),
      debug=kxm.Debug(
          catch_post_mortem=False,
      ),
      subdir_format=kxm.SubdirFormat(
          wu_dirname='{wid}{separator_if_sweep}{sweep_kwargs}',
      ),
      executor=xm_abc.Borg(
          logs_read_access_roles=['all'],
          # Autopilot has regularly caused us "pending forever" issues.
          autopilot_params=xm_abc.AutopilotParams(enabled=False),
          # This can help quite a bit for batch/freebie jobs:
          scheduling=xm_abc.BorgScheduling(
              max_task_failures=-1,
              max_per_task_failures=0,
              task_failure_credit_period=3600,
          ),
      ),
      # Activate tensorboard.corp
      add_tensorboard_corp=True,
  )
