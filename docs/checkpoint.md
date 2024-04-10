# Checkpoint

## Partial loading

To load weights from another checkpoint (e.g. restore pretrained encoder), you
can use the `init_transforms` argument of `kd.train.Trainer`

```python
cfg.init_transforms = {
    'pretrained_init': kd.ckpts.PartialKauldronLoader(
        workdir=kd.ckpts.workdir_from_xid(12345, wid=1),
        new_to_old={  # Mapping params
            # '<new_path>':            '<source_path>'
            'params.decoder.layers_0': 'params.endoder',
        },
    )
}

trainer = konfig.resolve(cfg)

# When initializing the weights, the `init_transform` is applied
init_state = trainer.init_state()

# `init_state.params['decoder']['layers_0']` now contains the previous encoder
# weights
```

See `kd.ckpts.PartialLoader` for details.
