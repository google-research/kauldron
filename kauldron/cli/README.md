# Kauldron CLI

Command-line interface for inspecting and debugging Kauldron configurations.

## Usage

Projects using `kauldron_binary` automatically get a `:trainer_cli` target:

```sh
kauldron <command> <sub_command> \
    --cfg=<path/to/config.py> [--cfg.override.key=value ...]
```

Call commands from Python as `kd.cli.<command>.<SubCommand>(cfg).execute()`.

### Example:

```sh
kauldron data element_spec \
    --cfg //third_party/py/kauldron/examples/mnist_autoencoder.py
```

and

```python
from kauldron import kd
kd.cli.data.ElementSpec(cfg).execute()
```

## List of Commands

* `config`
  - `show`: Print the unresolved config tree.
  - `resolve`: Resolve and print the fully-instantiated config.

* `data`
  - `element_spec`: Display the element spec of the training data pipeline.
