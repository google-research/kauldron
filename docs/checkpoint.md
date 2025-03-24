# Checkpoint

## Restoring a Checkpoint

To load weights from a particular checkpoint, use the checkpointer attribute
of the trainer:

```python
trainer = kd.from_xid.get_resolved(xid, wid=wid)

init_state = trainer.init_state()
state = trainer.checkpointer.restore(init_state, step=-1)
```

## Partial loading

To load weights from another checkpoint (e.g. restore pretrained encoder), you
can use the `init_transform` argument of `kd.train.Trainer`

```python
cfg.init_transform = kd.ckpts.PartialKauldronLoader(
    workdir=/path/to/old/workdir/,
    new_to_old={  # Mapping params
        # The new_to_old dict determines which weights are loaded from the
        # target checkpoint, and can also be used to rename subtrees when
        # loading params from a different pretrained model.
        # '<new_path>':            '<source_path>'
        'params.decoder.layers_0': 'params.endoder',
    },
)

trainer = konfig.resolve(cfg)

# When initializing the weights, the `init_transform` is applied
init_state = trainer.init_state()

# `init_state.params['decoder']['layers_0']` now contains the previous encoder
# weights
```

See `kd.ckpts.PartialLoader` for details.

## Relaunching an experiment

To relaunch an experiment, you can:

*   Continue training in a new separate workdir: Relaunch a new job with
    `init_transform`, to initialize your new model to the previous state:

    ```python
    cfg.init_transform = kd.ckpts.PartialKauldronLoader(
        workdir=/path/to/old/workdir/,
        new_to_old={  # Mapping params
            'step': 'step',
            'params': 'params',
            'collections': 'collections',
            'opt_state': 'opt_state',
        },
    )
    ```
