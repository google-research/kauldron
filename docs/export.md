# Model export

At the beginning of training, kauldron automatically serializes the model into
the workdir.
This is controlled by the `cfg.exporter` which defaults to a
`kd.export.JaxModelExporter`.

## JaxModelExporter
The `kd.export.JaxModelExporter` serializes a forward pass of the model using
the [`jax.export`](https://docs.jax.dev/en/latest/jax.export.html#module-jax.export) module (see [jax serialization docs](https://docs.jax.dev/en/latest/export/export.html)).

Essentially it constructs a forward function like this:

```python
def forward(*, params, key, **kwargs) -> dict[str, Any]:
  preds, collections = model.apply(
      {'params': params},
      rngs=_rng_streams_from_key(key),
      ...
      **kwargs)
  return {"preds": preds, "interms": interms}
```

where the `**kwargs` correspond to the inputs to the `model.__call__` method.
This function is then serialized and written to
`"{workdir}/train_model.jax_exported"`.

Loading the model again is pretty easy and does not depend on `kauldron`
(only on `jax` and `orbax`):

```python
import jax.export
from etils import epath
import orbax.checkpoint as ocp

WORKDIR = epath.Path("...")

PATH = WORKDIR / "train_model.jax_exported"
# Load the model
model = jax.export.deserialize(PATH.read_bytes())
# Load the params from the checkpoint
checkpointer = ocp.CheckpointManager(
  WORKDIR / "checkpoints",
  options=ocp.CheckpointManagerOptions(step_prefix="ckpt")
)
step = checkpointer.latest_step()
state = checkpointer.restore(step=step, args=ocp.args.StandardRestore())
params = state['params']

# Define a custom forward function that returns only the preds
# Here assuming the model inputs are called `image`
@jax.jit
def forward(image, params, key=jax.random.PRNGKey(0)):
  out = model.call(params=params, key=key, image=image)
  return out["preds"]


# Run the model on a custom input
forward(jnp.zeros((8, 64, 64, 3)), params)
```

NOTE: One important caveat is that the model has to be run with the same number
of devices as it was exported. The same restriction also applies to the
checkpoint loading above which is sensitive to differences in sharding.
Both of these issues can likely be resolved, but are currently limitations.
