# konfig

## Konfig best practices

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

### Colab and imports

On Colab, it's very easy to mix regular imports with `konfig.imports()`. To
avoid this issue:

* On Colab: **Never** use `konfig.imports()`. Instead, locally mock the modules
  with `konfig.mock_modules()`.

  ```python
  from flax import nn  # All imports are outside `konfig.imports()`
  import optax

  # Imports are mocked only locally
  with konfig.mock_modules():
    cfg.model = nn.Dense()
  ```

* Outside Colab: **Only** use `konfig.import()` to import configurables. This
  ensure clear boundaries between configurable modules and the config
  implementation.

  ```python
  with konfig.imports():
    from flax import nn
    import optax

  cfg.model = nn.Dense()
  ```
