# OO Random API

`kd.random` is a small wrapper around `jax.random` that is semantically equivalent. It provides:

*   Reduce boilerplate through an OO API.
*   Support `str` in `fold_in`
*   Full support with the Jax/Flax API.

Usage:

```python
key = kd.random.PRNGKey(0)

key0, key1 = key.split()  # ObjecO API

key1 = key1.fold_in('param')  # Fold-in supports `str`

# Those 2 lines are equivalent:
x = key0.uniform()
x = jax.random.uniform(key)  # Jax API still works
```

Note: The `random` is a self-contained standalone module that can be
imported in non-kauldron projects (`from kauldron import random`).