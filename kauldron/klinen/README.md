# Klinen - Torch-like API for flax

Klinen is a small wrapper around `flax.linen`. The goal is to provide a
stateless, object-oriented, supporting auto-complete and type checking.

## Documentation

### Model creation

Model creation is similar to flax, except the modules should inherit from
`klinen` instead of `linen`:

```python
from flax import linen as nn
from kauldron import klinen as knn


class MLP(knn.Module):

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(32)(x)


class AutoEncoder(knn.Module):
  encoder: knn.Module
  decoder: knn.Module

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return self.decoder(self.encoder(x))
```

* Inside `knn.Module`, any linen modules can be used.

Note: Alternatively, you can convert any existing `flax.linen.Module` to klinen
with `MyKlinenModule = knn.convert(MyFlaxModule)`

### Model initialization / usage

To initialize the model, use `model.init_bind()` instead of `model.init()`. It
will return a copy of the module with bind parameters.

```python
model = AutoEncoder(
    encoder=MLP(),
    decoder=MLP(),
)
model = model.init_bind(rng, jnp.zeros((batch_size, 64)))

# After the model is initialized, it can be called directly
y = model(x)

# You can also call individual sub-modules
y = model.encoder(x)
```

### Randomness

`klinen` modules are stateless, this mean they are fully deterministic. Calling
the same model twice will always return the same result. If your model uses
randomness (e.g. `nn.Dropout`), the `rng` key has to be explicitly provided:

```python
model = model.with_rng(rng)  # Replace the current rng.

y0 = model(x)
y1 = model(x)

assert jnp.allclose(y0, y1)  # Calling the model twice give the same output

model = model.with_rng(rng2)  # Set a new rng
```

Multiple values are accepted:

* `model.with_rng({'dropout': rng})`: Rng streams explicitly defined
* `model.with_rng(rng)`: Key distributed among streams (with
  `rng.fold_in(stream_name)`)
* `model.with_rng()`: no-argument provided, split the current `rng` to get the
  next one.

Calling `model(x)` before a key was provided with `.with_rng` will yield an
error the first time.

Currently, there's no guarantee that the encoder called in `model(x)` or
directly with `model.encoder(x)` have the same rng. This will be fixed in the
future.

### Training/eval mode

To disable determinism, models can use the `self.training` attribute:

```python
class MLP(knn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2)(x)
    x = nn.Dropout(0.5)(x, deterministic=not self.training)
    return x
```

By default, `model.training == True`. You can switch the model to eval mode
with `.eval()`

```python
model = model.eval()  # Switch to eval mode
assert not model.training


model = model.train()  # Switch back to train mode
assert model.training
```

### Parameters

You can access the flax parameters, either at the root level or for individual
modules.

```python
model.params
model.encoder.params
model.encoder.params['Dense_0']  # nn.Dense params defined inside `nn.compact`
```

### Jit, auto-diff

`knn.Module` are compatible with `jax.tree_utils` to map over the parameters.
This means modules can be used nativelly inside `jax.jit`:

```python
@jax.jit
def eval_step(model: knn.Model, x: jax.Array, y: jax.Array) -> jax.Array:
  model = model.eval()
  y_pred = model(x)
  return loss(y_pred, y)
```

### Intermediate values

Often, it's very convenient to be able to store/access intermediate values
in the module tree. It is possible by annotating module fields as
`knn.Intermediate[T] = dataclasses.field(init=False)`.

```python
class Sequential(knn.Module):
  childs: list[nn.Module]

  tmp_values: knn.Intermediate[list[jax.Array]] = dataclasses.field(
      init=False,
      default_factory=list,
  )

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    for child in childs:
      x = child(x)
      self.tmp_values.append(x)
    return x
```

The intermediate values are reset at each call (calling `model()` twice
will create a new `tmp_values` list). Intermediate values are not bound to
the model object. Instead they need to be explicitly fetched:

```python
model = AutoEncoder(
    encoder=Sequential([
        nn.Dense(32),
        nn.Dense(32),
    ]),
    decoder=MLP(),
)
model = model.init_bind(rng, x)

y = model(x)  # Standard call (no intermediate)

with model.capture_intermediates() as intermediates:
  y = model(x)

# Convenience wrapper around `model.capture_intermediates()`
y, intermediates = model.call_with_intermediate(x)


# `intermediates` has the same structure as the `model`, but only sub-modules
# and `knn.Intermediate` fields are available.
tmp_values = intermediates.encoder.tmp_values
```
