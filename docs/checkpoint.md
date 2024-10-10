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

## Relaunching an experiment

To relaunch an experiment, there's a few options possible:

*   Restart from the XM UI
*   Relaunch a new job with `init_transform`, to initialize your new model to
    the previous state:

    ```python
    cfg.init_transforms = {
        'pretrained_init': kd.ckpts.PartialKauldronLoader(
            workdir=kd.ckpts.workdir_from_xid(12345, wid=1),
            new_to_old={  # Mapping params
                'step': 'step',
                'params': 'params',
                'collections': 'collections',
                'opt_state': 'opt_state',
            },
        )
    }
    ```

    This will continue the training in a separate workdir.

*   Restart a new job while re-using the previous workdir. If your work-unit was
    launched in `/path/to/.../kd/<xid>/<wid>/`, you can relaunch it with:

    ```sh
    --xp.root_dir=/path/to/.../kd/ \
    --xp.subdir_format.xp_dirname=<xid> \
    --xp.subdir_format.wu_dirname=<wid>
    ```
