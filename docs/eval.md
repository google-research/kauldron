# Train, eval, randomness

## Evaluation

### Use eval

Eval can be defined on the `evals` attribute of `kd.train.Trainer`:

```python
cfg = kd.train.Trainer()
cfg.evals = {
    'eval': kd.evals.Evaluator(
        run=kd.evals.EveryNSteps(100),
        num_batches=None,
        ds=_make_ds(training=False),
        metrics={},
    )
}
```

If `kd.evals.Evaluator` does not define losses, metrics, summaries, those are
reused from train.

Where/how the `kd.evals.Evaluator` is run can be specified through the `run=`
kwargs. Evaluators can run:

*   Within the `train` job:
    *   `EveryNSteps`: Run evaluation every `X` steps
    *   `Once`: Run a single evaluation after `X` steps
*   In a separate XManager job:
    *   `StandaloneEveryCheckpoint`: Run evaluation every time a new checkpoint
        is found.
    *   `StandaloneLastCheckpoint`: Only run evaluation once, after train has
        completed.

Evaluators run in a standalone job can be grouped together through the
`job_group='group_name'` attribute. This allow to save resources by sharing the
same job for multiple evaluators.

The `StandaloneXxx` supports all `kxm.Job` parameters, if you need to run
evaluator on a different platform,...

See
[mnist_standalone_eval.py](https://github.com/google-research/kauldron/tree/main/examples/mnist_standalone_eval.py)
for an example.

When run as a standalone job, you can use different XManager options between the
train and eval jobs (defined both in the config or through flags):

*   `--xp.platform`: Set the value globally (for both train and eval)
*   `--cfg.xm_job.platform`: Set the value for train only
*   `--cfg.evals.<my-eval>.run.platform`: Set the value for eval only

Note: Using `--xp.platfom` and `--cfg.xxx.platform` are mutually exclusive!

### Start an eval-only job

Sometimes, you only want to run evaluation on a trainer from a previous
Kauldron experiment.
experiment. This can be achieved through `kd.train.Trainer.eval_only()`:

```python
def config():
  cfg = kd.train.Trainer.eval_only()
  cfg.evals = {
      'my_eval': kd.evals.Evaluator(
          run=kd.evals.StandaloneLastCheckpoint(),
          ...,
      ),
  }
  return cfg
```

See
[mnist_eval_only.py](https://github.com/google-research/kauldron/tree/main/examples/mnist_eval_only.py)
for an example.

Note: `kd.train.Trainer.eval_only()` only works when used inside `konfig`.

### Train / eval in Module

Model can detect if they are in training / eval mode by using the
`kd.nn.train_property`.

```python
class MyModel(nn.Module):
  # Create a `@property` that will look-up the global `is_training` value
  # when called
  is_training = kd.nn.train_property()  # No annotations here !!

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

Inside a module, you can overwrite the `is_training` value with the
`kd.nn.set_train_property` contextmanager:

```python
class MyModule(nn.Module):

  @nn.compact
  def __call__(self, x):
    with kd.nn.set_train_property(False):
      x = self.pretrained_encoder(x)
```

`kd.nn.set_train_property` can also be used to call a Kauldron model inside a
non-Kauldron model (to propagate the `train` / `deterministic` kwarg to the
model).

## Training

### Create the trainer

The root trainer object is `kd.train.Trainer` which defines the model, datasets,
metrics, losses,...

See
[mnist_autoencoder.py](https://github.com/google-research/kauldron/tree/main/examples/mnist_autoencoder.py)
for an example.

### High level API

The `Config` can be run by calling the `.train()` method. It will take care of
everything (checkpoint, eval, summaries,...).

```python
trainer.train()
```

### Mid level API

If you only need to run the training loop:

```python
state = trainer.init_state()

for batch in trainer.train_ds.device_put(trainer.sharding.ds):
  state, aux = trainer.trainstep.step(state, batch)
```

The `.device_put()` is chained with the dataset to put examples on devices (
default to `kd.sharding.FIRST_DIM`).

## Randomness

### Determinism

Kauldron uses a global seed (`trainer.seed = 42`) that is then split into the
various sub-components (dataset, model,...). For more control, the seed can also
be explicitly set inside the submodules (e.g. `trainer.train_ds.seed = 42`)

### Rng streams

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

```python
params = model.init(rng_streams.init_rngs(), ...)

@jax.jit
def forward(step, params, batch):
  rngs = rng_streams.train_rngs(step)  # Create the rng for current `step`
  return model.apply(params, batch, rngs=rngs)
```
