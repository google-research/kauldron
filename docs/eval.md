# Eval and determinism

## Use eval

Eval can be defined on the `eval` attribute of `kd.train.Trainer`:

```python
cfg = kd.train.Trainer()
cfg.evals = {
    'eval': kd.train.Evaluator(
        run_every=100,
        num_batches=None,
        ds=_make_ds(training=False),
        metrics={},
    )
}
```

If `kd.train.Evaluator` does not define losses, metrics, summaries, those are
reused from train.

## Train / eval in Module

Model can detect if they are in training / eval mode by using the
`kd.train.train_property`.

```python
class MyModel(nn.Module):
  # Create a `@property` that will look-up the global `is_training` value
  # when called
  is_training = kd.train.train_property()  # No annotations here !!

  @nn.compact
  def __call__(self, x):
    # Inside the methods, `self.is_training` can be called
    if self.is_training:
      rng = self.make_rng('default')
      x = jax.random.choice(rng, x)

    # `kd.nn.Dropout` supports `is_training` by default (no need to
    # propagate `deterministic=`)
    x = kd.nn.Dropout(0.5)(x)

    return x
```

The `self.is_training` value is set globally in `model.apply` / `model.init` for
all submodules. No more `deterministic` kwargs to propagate through your modules
!!

```python
model = MyModel()
params = model.init(..., is_training_property=True)

y = model.apply(..., is_training_property=False)  # Eval
```

## Rng streams

By default, the following `rng` streams are created:

*   `params`: Only during `.init()`
*   `dropout`: For `nn.Dropout`, only available in training (not eval).
*   `default`: Default `rng` stream, only available in training (not eval).

If you need custom streams, or need to overwrite the default values. You can set
the `rng_streams` attribute of `kd.train.Trainer` to `kd.train.RngStreams`. Note
that the `kd.train.RngStreams` will be **merged** with the default streams (so
you don't need to re-specify `params`,...):

```python
cfg = kd.train.Trainer()
cfg.rng_streams = kd.train.RngStreams([
    # Overwrite `dropout` stream to only be activated in `eval`
    kd.train.RngStream('dropout', train=False, eval=True),
    # Add a custom stream (by default only on `train`)
    kd.train.RngStream('my_custom_stream'),
])
```

To get the `{'dropout': rng, ...}` values, call the `rng_streams.train_rngs()`,
`.eval_rngs()` or `.init_rngs()`.

By default, the train and eval rngs require the `device` index from `@pmap`, so
each replicated computation uses a different rng.

```python
params = model.init(rng_streams.init_rngs(), ...)

@jax.jit
def forward(step, params, batch):
  rngs = rng_streams.train_rngs(step)  # Create the rng for current `step`
  return model.apply(params, batch, rngs=rngs)
```
