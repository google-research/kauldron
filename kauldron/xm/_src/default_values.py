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

"""Default values and configuration.

This file is shared by both `kxm` and `kd`.
"""

from kauldron import konfig

with konfig.imports(lazy=True):
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from kauldron import kd
  from kauldron import kxm
  from xmanager import xm
  from xmanager import xm_abc
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


# TODO(epot): Could automatically deduce which fields should be
# initialized as `dict`/subclass using dataclasses.field().default_factory
# but tricky in practice (not all subclasses should be created).
# TODO(epot): Add a `__konfig_default_init__` to move those directly in the
# files where they are used. Currently not possible as it would break
# `__qualname__`

konfig.register_default_values(
    xm_abc.Borg(
        scheduling=xm_abc.BorgScheduling(),
        autopilot_params=xm_abc.AutopilotParams(),
        borglet_params=xm_abc.BorgletParams(),
    )
)

konfig.register_default_values(
    xm_abc.Borg(
        scheduling=xm_abc.BorgScheduling(),
        autopilot_params=xm_abc.AutopilotParams(),
        borglet_params=xm_abc.BorgletParams(),
    )
)
konfig.register_default_values(
    kxm.Job(
        args={},
        env_vars={},
        files={},
        requirements=xm.JobRequirements(),
        executor=xm_abc.Borg(),
        debug=kxm.Debug(),
        interpreter_info=kxm.InterpreterInfo(),
    ),
)
konfig.register_default_values(
    kxm.Experiment(
        # Use `object` to support both `--xp.sweep` and `--xp.sweep=lr,batch`
        sweep=konfig.placeholder(object),
        # JobParams attributes
        args={},
        env_vars={},
        files={},
        requirements=xm.JobRequirements(),
        executor=xm_abc.Borg(),
        debug=kxm.Debug(),
        interpreter_info=kxm.InterpreterInfo(),
        # Experiment attributes
        # tags=konfig.placeholder(object), ?
        subdir_format=kxm.SubdirFormat(),
        execution_settings=xm_abc.ExecutionSettings(),
    )
)

konfig.register_default_values(
    kd.train.Trainer(
        workdir=konfig.placeholder(str),
        evals={},
        xm_job=kxm.Job(),
    )
)

# Register aliases for cleaner config display
konfig.register_aliases({
    'kauldron.kd': 'kd',
    'kauldron.kxm': 'kxm',
    # TODO(epot): Support pattern `'kauldron.projects.$0': '$0'`
    # 'kauldron.projects.$0': '$0',
})
