# NNX modules

Kauldron natively uses
[Flax Linen](https://flax.readthedocs.io/en/latest/linen_intro/index.html)
modules. If your model is written with the newer
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) API, you can
use it in Kauldron by creating a module that inherits from
`kd.contrib.knnx.KdNnxModule` and initializes the nnx module in a `setup`
method.

## Usage

Because we don't want to create the Nnx module during config resolution, we use
the following syntax to create a knnx module:

```python
from kauldron import kd
import dataclasses

@dataclasses.dataclass(kw_only=True)
class MyKdNnxModule(kd.contrib.knnx.KdNnxModule):
  input_dim: int = 3
  hdim: int = 10

  image: kontext.Key = "batch.image"

  def setup(self, rngs: nnx.Rngs=nnx.Rngs(0)):
    self.lin = nnx.Linear(self.input_dim, self.hdim, rngs=rngs)

  def __call__(self, image):
    return self.lin(image)
```

Then the module can be used either as a Nnx module:

```python
mod = MyModule()
mod.setup()
out = mod(jnp.ones(1, 32, 32, 3))
```

Or as a flax linen module:

```python
mod = MyModule()
vars = mod.init({'params':jax.random.key(0)}, jnp.ones(1, 32, 32, 3))
out = mod.apply(vars, rngs={})
```

Note: To capture intermediates, use the following syntax in your NNX module:

```python
class MyModel(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(2, 5, rngs=rngs)

    def __call__(self, x):
        x = self.linear(x)
        self.sow(nnx.Intermediate, 'h1', x)  # Sowing here
        return x
```

### Hierarchical NNX modules

It is discouraged to put NNX modules in kauldron's configs, because these will
be created during config resolution and we actually want to delay module
creation to the`init` phase. It's recommended to use partial functions in the
config like the following:

In your config:

```python
with konfig.imports():
  from functools import partial
  from ... import my_module

def get_config():
  cfg = Trainer()
  cfg.model = my_module.MainModule(
  backbone_init=partial(my_module.NNXBackbone, arg1=value1))
  ...
  return cfg
```

In your my_module file:

```python
from flax import nnx
from kauldron.contrib import knnx
import typing as tp

class NNXBackbone(nnx.Module):
  def __init__(self, arg1):
    self.arg1 = value1

# main kauldron module
@dataclasses.dataclass(kw_only=True)
class MyModule(knnx.KdNnxModule):
  backbone_init: tp.Callable

  def setup(self):
    self.backbone = self.backbone_init()

  def __call__(self, x):
    out = self.backbone(x)
    ...
```

## How it works

Under the hood, the module is created by calling the `init` method, which
instantiates the NNX module by calling the `setup` method. It then uses
`nnx.split` / `nnx.merge` to store the NNX module's graph definition,
parameters, and state into Linen's variable scope.

*   At **init** time: the NNX module is instantiated and its state is split into
    Linen variables.
*   At **apply** time: the state is merged back into an NNX module, a forward
    pass is made, and the updated state is written back.

### Train / eval mode

The wrapped module automatically calls `.train()` / `.eval()` on the NNX module
based on the variable `is_training_property` passed by kauldron to the `apply`
function.

## Legacy wrapper documentation

Kauldron natively uses
[Flax Linen](https://flax.readthedocs.io/en/latest/linen_intro/index.html)
modules. If your model is written with the newer
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) API, you can
use it in Kauldron by wrapping it with `kd.contrib.nn.linen_from_nnx`.

### Usage

Define your NNX module as usual:

```python
from flax import nnx


class MyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(3, 4, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)
```

Then wrap it with `linen_from_nnx` in your config:

```python
cfg.model = kd.contrib.nn.linen_from_nnx(MyModel)
```

`linen_from_nnx` wraps the NNX module into a Linen module, so it integrates with
Kauldron's `TrainStep`, checkpointing, and all other components.

Constructor arguments can be passed directly:

```python
cfg.model = kd_nn.linen_from_nnx(nnx.Linear, in_features=3, out_features=4)
```

### How it works

Under the hood, `linen_from_nnx` delays instantiation of the NNX module to the
Linen `init` phase. It then uses `nnx.split` / `nnx.merge` to store the NNX
module's graph definition, parameters, and state into Linen's variable scope.

*   At **init** time: the NNX module is instantiated and its state is split into
    Linen variables.
*   At **apply** time: the state is merged back into an NNX module, a forward
    pass is made, and the updated state is written back.

### Hierarchical NNX modules

`linen_from_nnx` supports hierarchical NNX modules. If your NNX module takes
other NNX modules as arguments, wrap each one with `linen_from_nnx`:

```python
cfg.model = kd_nn.linen_from_nnx(
    MyEncoder,
    backbone=kd_nn.linen_from_nnx(MyBackbone, hidden_dim=64),
)
```

The sub-modules will be recursively instantiated at init time.

### Train / eval mode

The wrapped module automatically calls `.train()` / `.eval()` on the NNX module
based on the Kauldron training context (see https://kauldron.rtfd.io/en/latest/eval.html#train-eval-in-module).

### Limitations

`kontext.Key` annotations on the NNX module are **not** supported.
`linen_from_nnx` wraps the NNX module inside a Linen shell, and `kontext`
inspects the wrapper class — not the inner NNX class. The NNX module receives
its inputs as regular positional arguments to `__call__`.
