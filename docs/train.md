# Training

## Create the trainer

The root trainer object is `kd.train.Config` which defines the model, datasets,
metrics, losses,...

See https://github.com/google-research/kauldron/tree/HEAD/kauldron/projects/examples/mnist_autoencoder.py for
an example.

## High level API

The `Config` can be run by calling the `.train()` method. It will take care of
everything (checkpoint, eval, summaries,...).

```python
cfg.train()
```

## Mid level API

If you only need to run the training loop:

```python
state = cfg.init_state()

for batch in cfg.train_ds:
  state, aux = cfg.trainstep.step(state, batch)
```
