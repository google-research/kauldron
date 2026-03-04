# konfig

## Modular configs with ConfigArgs

ConfigArgs is a pattern to declare main configuration hyper-parameters that
you can use to create modular configs.

Basic example in a config file:

```python
_DATASET_DIMS = {'mnist': 10, 'imagenet': 1000}

@dataclasses.dataclass(kw_only=True, frozen=True)
class ConfigArgs:
  dataset_name: str = 'mnist'
  model_name = 'vit'


def get_config(args: ConfigArgs = ConfigArgs()):
  cfg = kd.train.Trainer()
  if args.model_name == 'vit':
    cfg.model = MyModel(dim=_DATASET_DIMS[args.dataset_name])
  else:
    ...
```

The values in `args` can be used to create a modular config.
ConfigArgs arguments can be set with the command-line with the following
pattern: `--cfg.__args__.arg1 = value`.

Note: By design we chose to disable sub-field overrides like
`--cfg.__args__.dict1.arg1 = value1` to avoid cluttering the ConfigArgs.

Sweeping on ConfigArgs values is also possible via

```python
def sweep_cfgarg():
  for dataset_name in ["mnist", "imagenet"]:
    yield {"__args__.dataset_name": dataset_name}
```

A detailed example can be found in the [ConfigArgs demo config](https://github.com/google-research/kauldron/tree/main/examples/configargs_demo.py).

### Good practices

* The `args`-dependent logic should be kept as simple as possible, E.g. only if
tests on hyper-parameters and for loops. More complex logic should be put
in the python objects (See).

* If a hyper-parameter value is used in multiple places in the config (e.g
affecting both model and dataset), you can put it in the ConfigArgs instead of
using the `cfg.ref.aux` pattern.

* If a field is used only once in the config, you should probably not put it in
ConfigArgs and mutate it through its regular config path.

### Comparison with previous modular system
Before ConfigArgs, there was two ways to create a modular config:

* the get_config could take a single string argument, which needed to be parsed
to create a modular config, which does not give the right level of control;

* the `cfg.aux` field of the config was used to store root-level hyper-
parameters, which were used as references with `cfg.ref.aux`. Since references
could not be used as the python object they point to,
they had to be sent to functions wrapped in `konfig.ref_fn`.
The ConfigArgs pattern should allow to remove most field references if not all
(if you have a use case that you think does not work, please reach out to us).

## Konfig best practices

Note: Please instead read the which explains best practices.

### Keep config code separated

Because `konfig` and standard Python code looks similar some conventions should
be followed to avoid mixing the two.

* All config code should go **inside** the `configs/` folder. If you need to
  split your config in multiple files, those should be imported **outside** the
  `konfig.imports()` context.
* All the actual implementation (models, dataset implementation, additional
  logic,...) should go **outside** the `configs/` folder. Those are imported
  **inside** the `konfig.imports()` context.

```python
from my_project.configs import base_config
from my_project.configs import ds_config

with konfig.imports():
  from my_project import my_ds


def get_config():
  cfg = base_config.get_config()
  cfg.train_ds = ds_config.make_ds(split='train')  # This returns a `ConfigDict`
  cfg.eval_ds = my_ds.MyDataset()  # This IS a `ConfigDict`
```

Note: If you have config files inside `configs/` that do not contain
`get_config()` (e.g. helpers containing only part of the config), you
need to exclude them from the config test through
`kauldron_binary(config_srcs_exclude=["configs/my_helper.py"])`.
