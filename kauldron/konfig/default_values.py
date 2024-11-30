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

This file do not add any overhead as it do not import anything.

Registering default values allows to pre-fill nested values, making them
available through CLI. For example, creating an empty `kxm.Experiment()`
automatically adds `--xp.requirements.cpu=`,
`--xp.executor.autopilot_params.enabled=`... without having to explicitly
define `cfg.executor = `, `cfg.requirements = `.
"""

from kauldron import konfig

with konfig.imports(lazy=True):
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from kauldron import kd
  from kauldron import kxm
  from xmanager import xm
  from xmanager import xm_abc
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


# We cannot automatically deduce which fields should be
# initialized as `dict`/subclass using dataclasses.field().default_factory
# as not all subclasses should be created. Better to be explicit.
# We cannot add a `__konfig_default_init__` to move those directly in the
# files where they are used. It would break `__qualname__`. And when imports
# are lazy, `__konfig_default_init__` would not be available.

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

# Resolving `kd.evals.StandaloneXxx` will trigger the `default_factories`, thus
# importing XM. To avoid this, we prefil the values with empty ConfigDict.
# Those won't be resolved due to `Standalone.__konfig_resolve_exclude_fields__`
konfig.register_default_values(
    kd.evals.StandaloneLastCheckpoint(
        requirements=xm.JobRequirements(),
        executor=xm_abc.Borg(),
    )
)
konfig.register_default_values(
    kd.evals.StandaloneEveryCheckpoint(
        requirements=xm.JobRequirements(),
        executor=xm_abc.Borg(),
    )
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
        schedules={},
        setup=kd.train.Setup(),
        xm_job=kxm.Job(),
    )
)

_eval_only_trainer = kd.train.Trainer.eval_only(
    workdir=konfig.placeholder(str),
    aux={
        'xid': konfig.placeholder(int),
        'wid': 1,
    },
    evals={},
    num_train_steps=0,
    stop_after_steps=0,
    optimizer=None,
    # No checkpointer. The weights are restored through `init_transform`.
    checkpointer=None,
    setup=kd.train.Setup(eval_only=True),
    xm_job=kxm.Job(),
)
_eval_only_trainer.update(  # pytype: disable=attribute-error
    # No train dataset, but specs are needed to initialize the model.
    train_ds=kd.data.ElementSpecDataset(
        spec=kd.from_xid.get_element_spec(
            xid=_eval_only_trainer.ref.aux['xid'],
            wid=_eval_only_trainer.ref.aux['wid'],
        ),
    ),
    model=kd.from_xid.get_resolved(
        xid=_eval_only_trainer.ref.aux['xid'],
        wid=_eval_only_trainer.ref.aux['wid'],
        path='model',
    ),
    init_transform=kd.ckpts.PartialKauldronLoader(
        workdir=kd.ckpts.workdir_from_xid(
            xid=_eval_only_trainer.ref.aux['xid'],
            wid=_eval_only_trainer.ref.aux['wid'],
        ),
        new_to_old={
            # Also restore the step so metrics are reported correctly.
            'step': 'step',
            'params': 'params',
            'collections': 'collections',
        },
    ),
    # TODO(epot): If present in the original config, should also re-use
    # `rng_streams`, sharding` !!
)

konfig.register_default_values(_eval_only_trainer)

# Register aliases for cleaner config display
konfig.register_aliases({
    'kauldron.kd': 'kd',
    'kauldron.kxm': 'kxm',
    # TODO(epot): Support pattern `'kauldron.projects.$0': '$0'`
    # 'kauldron.projects.$0': '$0',
})
