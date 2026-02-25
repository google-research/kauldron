# NNX modules

Kauldron natively uses
[Flax Linen](https://flax.readthedocs.io/en/latest/linen_intro/index.html)
modules. If your model is written with the newer
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) API, you can
use it in Kauldron by wrapping it with `kd.contrib.nn.linen_from_nnx`.

## Usage

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

## How it works

Under the hood, `linen_from_nnx` delays instantiation of the NNX module to the
Linen `init` phase. It then uses `nnx.split` / `nnx.merge` to store the NNX
module's graph definition, parameters, and state into Linen's variable scope.

*   At **init** time: the NNX module is instantiated and its state is split into
    Linen variables.
*   At **apply** time: the state is merged back into an NNX module, a forward
    pass is made, and the updated state is written back.

## Hierarchical NNX modules

`linen_from_nnx` supports hierarchical NNX modules. If your NNX module takes
other NNX modules as arguments, wrap each one with `linen_from_nnx`:

```python
cfg.model = kd_nn.linen_from_nnx(
    MyEncoder,
    backbone=kd_nn.linen_from_nnx(MyBackbone, hidden_dim=64),
)
```

The sub-modules will be recursively instantiated at init time.

## Train / eval mode

The wrapped module automatically calls `.train()` / `.eval()` on the NNX module
based on the Kauldron training context (see https://kauldron.rtfd.io/en/latest/eval.html#train-eval-in-module).

## Limitations

`kontext.Key` annotations on the NNX module are **not** supported.
`linen_from_nnx` wraps the NNX module inside a Linen shell, and `kontext`
inspects the wrapper class — not the inner NNX class. The NNX module receives
its inputs as regular positional arguments to `__call__`.
