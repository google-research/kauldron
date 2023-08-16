# Checkpoint

## Partial loading

To load weights from another checkpoint (e.g. restore pretrained encoder), you
can use the `partial_initializer` argument of `kd.train.Checkpoint`

```python
checkpoint = kd.train.Checkpoint(
    partial_initializer=kd.ckpts.PartialLoader(
        source=kd.ckpts.KauldronSource('/path/to/original/work_unit/'),
        # Mapping params from <original state> -> <new state>
        new_to_old={
            'params.decoder.layers_0': 'params.endoder',
        },
    )
)

# If the checkpoint does not exists, the `partial_initializer` is used to
# initialize the weights
init_state = checkpoint.restore(init_state)

# `init_state.params['decoder']['layers_0']` now contains the previous encoder
# weights
```

See `kd.ckpts.PartialLoader` for details.
