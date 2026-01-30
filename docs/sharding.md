# Sharding & Model parallelism

https://kauldron.rtfd.io/en/latest/sharding.html

[TOC]

## Model parallelism

Sharding is defined through the `trainer.sharding` attribute.

```python
cfg.sharding = kd.sharding.ShardingStrategy(
    params={
        'encoder': kd.sharding.REPLICATED,
        'decoder': my_project.my_sharding_strategy,
    },
    opt_state=None,  # Let jax auto-infer the sharding
)
```

Each leaf of the sharding pytree can be:

*   `None`: Sharding is auto-inferred by jax (propagated from the inputs)
*   `jax.sharding.Sharding`: Explicitly set the sharding
*   `Callable`: Lazily compute the sharding from the array sub-tree:

    ```python
    def my_sharding_strategy(params: PyTree[jax.Array]) -> kd.sharding.ShardingTree:
      devices = np.asarray(jax.devices())
      devices = devices.reshape((-1, jax.device_count() // 4))
      mesh = jax.sharding.Mesh(devices, axis_names=('data', 'params'))

      def _shard_param(path, x):
        if 'kernel' in path:
          return jax.sharding.NamedSharding(
              mesh, jax.sharding.PartitionSpec('params')
          )
        elif 'bias' in path:
          return kd.sharding.REPLICATED
        else:
          raise ValueError(f'Unexpected param: {path}: {x.shape}')

      return tree.map_structure_with_path(_shard_param, params)
    ```

## Available sharding

By default, Kauldron provides the following `jax.sharding.Sharding`:

*   `kd.sharding.REPLICATED`: All devices hold the same data. This is the
    default `params` sharding.
*   `kd.sharding.FIRST_DIM`: First dimension is sharded across devices. This is
    the default dataset sharding.

Sharding is applied internally with `kd.sharding.with_sharding_constraint`:

```python
@jax.jit
def _step(state: TrainState, batch) -> TrainState:
  ...
  return kd.sharding.with_sharding_constraint(state, trainer.sharding.state)


for ex in trainer.train_ds.device_put(trainer.sharding.batch):
  state = _step(state, ex)
```
