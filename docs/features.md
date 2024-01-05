# Features

## Simpler random API

Kauldron expose a `random` module. It's a small wrapper around `jax.random` that
is semantically equivalent, but reduce boilerplate through an OO API.

Usage:

```python
key = kd.random.PRNGKey(0)

key0, key1 = key.split()  # ObjecO API

key1 = key1.fold_in('param')  # Fold-in supports `str`

# Those 2 lines are equivalent:
x = key0.uniform()
x = jax.random.uniform(key)  # Jax API still works
```

Note: The `random` is a self-contained standalone module that can be imported in
non-kauldron projects (`from kauldron import random`).

## Advanced sweeps

If your config defines multiple sweeps, you can create multiple `sweep_[NAME]()`
functions. In the launcher, you can use `--xp.sweep_info.names=aaa,bbb` to run
the **product** of `sweep_aaa()` and `sweep_bbb()`.

`--xp.sweep_info.names=name` specifies **all** sweeps to launch, so launching
the unnamed `def sweep()` require to add an empty `,`. Examples:

*   `--xp.sweep` launches `sweep()`
*   `--xp.sweep_info.names=lr` launches `sweep_lr()`
*   `--xp.sweep_info.names=lr,` launches the product of `sweep_lr()`, `sweep()`
    (notice the trailing `,`)
*   `--xp.sweep_info.names=lr,batch` launches the product of `sweep_lr()`,
    `sweep_batch()`

<!--

TODO(epot): Add more docs on the various sub-componenents

## Kauldron core

Kauldron is composed of individual self-contained sub-modules, which can be
imported and used independently:

Can be used by other codebase:

*   `kd.konfig`:
*   `kd.random`:
*   `kd.typing`:
*   `kd.klinen`:

Core features:

*   `kd.data`:
*   `kd.train`:
*   `kd.chkpt`:
*   `kd.inspect`:

Training utils:

Model, metrics,...:

*   `kd.nn`:
*   `kd.metrics`:
*   `kd.losses`:
*   `kd.summaries`:

-->
