# Introduction

Kauldron is a research codebase optimized for **research velocity** and
**modularity**:

The codebase itself is divided between:

*   Kauldron core: Collection of composable modules and tools providing all the
    boilerplate required for research and run experiments.
*   Kauldron projects: Individual research projects using the Kauldron core.
    While this is not a requirement, by convention, projects are located in
    [`kauldron/projects/`](http://https://github.com/google-research/kauldron/tree/HEAD/kauldron/projects/).

## Define experiment

### Config system

At the core of Kauldron is the config system. Each user compose their experiment
in their `config.py` by choosing which sub-modules to use. The root of the
config is the `kd.train.Config` trainer object which defines which model,
dataset, metrics,... to use.

The config objects looks like standard Python call (which allow auto-complete /
type checking), but instead each function call create a `ConfigDict` object
(based on `ml_collections`). This works through the magic `konfig.imports()`
contextmanager:

```python
with konfig.imports():
  import optax  # This looks like optax, but instead is a ConfigDict builder


cfg = optax.adam(learning_rate=0.003)  # This create a ConfigDict object !!!

assert cfg == konfig.ConfigDict({
    '__qualname__': 'optax:adam',
    'learning_rate'  == 0.003,
})

cfg.learning_rate = 1e-4  # Config can be mutated

optimizer = konfig.resolve(cfg)  # Resolve the actual object (here optax.adam)
```

Note: The `konfig` is a self-contained standalone module that can be imported
outside in non-kauldron projects (`from kauldron import konfig`).

<!--

TODO(epot): Add:

* `cfg.ref`
* Schedules & optimizers
* Detail each part of the config: dataset, evaluation, partial loading,...

-->

### Config references

Sometimes, a config value is reused in multiple places. When this happen, the
config should make sure updating one value update all places. This is done by
using reference `cfg.ref.entry` instead of `cfg.entry`.

For example:

```python
cfg.num_train_steps = 10_000

cfg.schedules = {
    "learning_rate": optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.001,
        decay_steps=cfg.ref.num_train_steps,  # <<< `.ref.` used here !!!!!
    )
}
```

When `num_train_steps` is changed (e.g. with `--config.num_train_steps=XX`), the
schedule will automatically adjust to the new value.

`kd.train.Config` defines a `cfg.aux = {}` field to store additional variables
that can easily be referred too through `cfg.ref.aux.xxx`, and globally updated.

### Keys and Context

Connecting elements together (batch to model, losses, metrics) is done through
string identifiers (called `keys`).

```python
cfg.model = AutoEncoder(input="batch.image")

cfg.train_loss = kd.losses.L2(
    preds="preds.image",
    targets="batch.image",
)
```

When executing the config, Kauldron will detect the keys and pass the matching
objects. In this example, Kauldron would run something like:

```python
for batch in ds:

  preds = model(input=batch['image'])

  loss = train_loss(preds=preds['image'], targets=batch['image'])
```

Each key starts by a registered prefix. Common prefixes includes:

*   `batch`: The output of the dataset (after all transformations)
*   `preds`: The output of the model.
*   `params`: Model parameters
*   `interms`: Flax intermediate variables (from `model.apply(...,
    capture_intermediates=True)`)

See http://https://github.com/google-research/kauldron/tree/HEAD/kauldron/utils/context.py for the full list of
identifiers.

Note: Keys can be arbitrary nested (e.g. `preds.cameras[0].pos`). `a.b` can
match both `a['b']` (index) or `a.b` (attribute).

The key system makes it very easy to add metrics, losses on arbitrary variables
(e.g. intermediate model output).

Internally, object using keys define them through the `: kontext.Key`
annotation. When the object is used, Kauldron will extract the actual value from
the keys and forward them to the method call.

```python
from kauldron import kontext

class AutoEncoder(flax.linen.Module):
  input: kontext.Key  # Key names match the `__call__` signature

  @nn.compact
  def __call__(self, input: Float["*b h w c"]) -> Float["*b h w c"]:
    ...
```
