# XManager launcher



Kauldron XManager launcher is a self-contained, independent library to launch
XManager jobs. It is designed to be:

* Generic: Can be used in any projects to launch any jobs, (not just
  Kauldron ones)
* Modular: Everything can be customized through the deep-configuration system
* Interactive: Colab friendly

## tl;dr;

If you're trying to launch a Kauldron experiment, you can use:

```sh
xmanager launch kauldron/xm/launch.py -- \
  --cfg=kauldron/examples/mnist_autoencoder.py \
  --cfg.train_ds.batch_size=32 \
  --xp.sweep=True \
  --xp.platform=jf=2x2
```

See for a list of flags.

## Usage

### Direct API

`kxm` experiments are entirely defined through the root `kxm.Experiment` object.
For example, here is a minimal experiment:

```python
from kauldron import kxm

xp = kxm.Experiment(
    jobs={
        'train': kxm.Job(
            target='//path/to/my:trainer'
            platform='jf=2x2',
            args={
                'workdir': kxm.WU_DIR_PROXY,
            },
        )
    },
    root_dir='/path/to/home/{author}/experiments/',
)
xp.launch()
```

Job parameters (`cell`, `platform`,...) can be set:

* At the individual `kd.Job` level
* At the `kxm.Experiment` level (applied to all jobs as fallback value)

### CLI (Config based)

While the direct API can be used for simplest use-cases, the main way to use
kxm is to wrap the `kxm.Experiment` inside `def get_config()` config file.
This allow better re-usability & overwriting any parameters through CLI.

```python
from kauldron import konfig

with konfig.imports():
  from kauldron import kxm

def get_config() -> kxm.Experiment:
  return kxm.Experiment(
      ...
  )
```

The config can then be launched with the default `kxm` launcher, which allow to
overwrite any fields:

```sh
xmanager launch kauldron/xm/launch.py -- \
  --xp=path/to/my/xp_config.py \
  --xp.jobs.train.platform=df=4x4 \
  --xp.jobs.eval.platform=cpu \
  --xp.cell=jn \
  --xp.note="My experiment description"
```

See to learn more about the config system.

### CLI (Existing configs)

`kxm` provides predefined configs that can be used directly or used as example:

<section class="zippy">

Launch a single binary:

```sh
xmanager launch kauldron/xm/launch.py -- \
  --xp=kauldron/xm/configs/single.py \
  --xp.target=//path/to/my:binary
```

</section>

<section class="zippy">

Launch a Kauldron experiment:

```sh
xmanager launch kauldron/xm/launch.py -- \
  --cfg=kauldron/examples/mnist_autoencoder.py \
  --cfg.train_ds.batch_size=32 \
  --xp.sweep=True \
  --xp.platform=jf=2x2
```

Kauldron experiments uses two ConfigDict flags:

* `--xp`: `kxm.Experiment` options (cell, platform, sweep,...)
* `--cfg`: Kauldron experiment to run (dataset, batch size,...)

Note that `--xp=kauldron/xm/configs/kd_single.py` is the default
value, so can be omitted for convenience.

</section>

### Colab

`kxm` can be used directly from Colab. Either through the direct API or by
importing a config.

See https://kauldron.rtfd.io/en/latest-xm colab to launch Kauldron experiments from Colab.

## How to

### Sweep

Note: If you're looking into launch sweep for Kauldron, see https://kauldron.rtfd.io/en/latest/intro.html#sweeps

Sweeps are defined through the `xp.sweep_info` attribute. `SweepInfo` customize
how to resolve the sweep. Different implementations are provided:

* `SimpleSweep`: Explicitly provide the list of args overwrite
  (`[{'batch_size': 64}, ...]`)
* `SweepFromCfg`: Load the `def sweep()` from the `config.py` (passed
  through `--cfg=path/to/cfg.py`)
  (`[{'batch_size': 64}, ...]`). Flags yield by the sweep functions are passed
  as-is to the binary.
* `KauldronSweep`: Load the sweep from the `def sweep()` function from the
  `config.py`. Values yield are merged with the `kd.train.Trainer` config.

Example:

```python
def sweep_fn():
  for batch_size in [32, 64]:
    yield {'batch_size': batch_size}

xp = kxm.Experiment(
    ...,
    sweep_info=kxm.SimpleSweep(sweep_fn),  # Jobs launched with `--batch_size=`
)
xp.launch()
```

For more flexibility, you can implement your own `SweepInfo`.

### Set workdir

The root work-unit and experiment directories are set at the top level, as it
is used in multiple places (both in TensorBoard, and individual jobs).

The experiment and work-unit directories are dynamically computed:

* `root_dir`: Root directory (provided by the user) containing all experiments
* `xp_dir`: Experiment directory (default to `{root_dir}/{xid}/`)
* `wu_dir`: Work-unit directory (default to `{root_dir}/{xid}/{wid}/`)

To pass the work-unit dir as argument to the binary, use `kxm.WU_DIR_PROXY`.
It returns a proxy string that will later be resolved (once all xid, wid,...
are known).

```python
xp = kxm.Experiment(
    # The root dir is defined at the `Experiment` level
    root_dir='/path/to/home/{author}/kd/'
    jobs={
        'train': kxm.Job(
            # A proxy string is used, that will later be resolved as
            # --workdir="/path/to/home/{author}/kd/{xid}/{wid}/"
            args={'workdir': kxm.WU_DIR_PROXY},
        )
    },
)
```

The experiment and work-unit directory names are customizable with the
`subdir_format` attribute:

```python
xp = kxm.Experiment(
    subdir_format=kxm.SubdirFormat(
        xp_dirname='{xid}-{name}',  # Default to `{xid}`
        wu_dirname='{wid}-{sweep_kwargs}',  # Default to `{wid}`
    ),
)
```

The `{}` will be automatically replaced by their matching values. The following
are supported:

* `{cell}`: Cell from the main job
* `{name}` / `{title}`: Experiment title
* `{user}` / `{author}`: Author who launched the experiment
* `{xid}`: Experiment id
* `{wid}`: Work-unit id, padded by the number of work-unit (e.g. `001`,
  `002`,... for 345 work-units)
* `{unpadded_wid}`: Work-unit id, without padding (`1`, `2`,...)
* `{sweep_kwargs}`: String representation of the sweep (e.g.
  `batch_size=32,seed=0`)
* `{separator_if_sweep}`: Simple `-` token only present if there's a sweep,
  used as separator (e.g.
  `wu_dirname='{wid}{separator_if_sweep}{sweep_kwargs}'`)
* Your custom `{}`, by subclassing `kxm.SubdirFormat`.

### Files dependencies

Similarly to work-unit directories, file dependencies are resolved using
(dynamically generated) proxy strings:

```python
kxm.Job(
    args={
        # Value dynamically resolved later
        'gin_config': kxm.file_path('my_file.gin'),
    },
    # Files to package with the MPM
    files={
        'my_file.gin': '//path/to/file.gin'
    },
)
```

### Config

If using the system, you can add `kxm.CFG_FLAG_VALUES` to your job
args:

```python
kxm.Job(
    args={
        'cfg': kxm.CFG_FLAG_VALUES,
    },
)
```

This allow to propagate the `--cfg.xxx` flags values from the `launch.py` to
your binary.

```sh
xmanager launch kauldron/xm/launch.py -- \
  --cfg=path/to/my_config.py \
  --cfg.train_ds.batch_size=32 \
  --xp=path/to/xm/launch_config.py \
  --xp.sweep=True \
  --xp.platform=jf=2x2
```

## Structure

Experiments are entirely defined through a unique root object:

* **`Experiment`**: Root object containing all experiment information (what to
  build, launch,...).

Experiment behavior can be customized through the following base classes:

* **`JobsProvider`** (`xp.jobs_provider`): Alternative way to provide the
  `dict[str, Job]` of jobs to run.
  Implementations include:
  * `KauldronJobs`: Extract the jobs dict from a Kauldron `config.py`
* **`SweepInfo`** (`xp.sweep_info`): Specify which sweep to run. Implementations include:
  * `SimpleSweep`: Explicitly provide the list of args overwrite
    (`[{'batch_size': 64}, ...]`)
  * `KauldronSweep`: Load the sweep from the `def sweep()` function from the
    `config.py`
* **`Orchestrator`** (`xp.orchestrator`): Schedule the jobs and work-unit. Can decide which jobs
  the sweep should be applied to, run jobs in a specific order, add vizier
  search,... For most use-cases, the default implementation should be sufficient.
* **`SubdirFormat`** (`xp.subdir_format`): Customize the experiment and
  work-unit directory name.

Other (non-overwritable objects)

* `Job`: Job contains all info about a single job (build target, platform,
  cell,...)
