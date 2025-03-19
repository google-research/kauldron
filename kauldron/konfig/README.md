# konfig

## Konfig best practices

Note: Please instead read the which explains best practices.

### Keep config code separated

Because `konfig` and standard Python code looks similar some conventions should
be followed to avoid mixing the two.

* All config code should go **inside** the `configs/` folder. If you need to
  split your config in multiple files, those should be imported **outside** the
  `kontext.imports()` context.
* All the actual implementation (models, dataset implementation, additional
  logic,...) should go **outside** the `configs/` folder. Those are imported
  **inside** the `kontext.imports()` context.

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
