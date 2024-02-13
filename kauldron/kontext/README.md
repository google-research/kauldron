# kontext

Kontext is a small self-contained library to manipulate nested trees.

Kontext introduce 2 main concepts: **Paths** and **Keys**.

## Paths

A `Path` is a string pointing to a nested tree value (e.g. `'a[0].b.inner'`).
Paths are used for tree manipulation/extraction:

```python
tree = {
    'a': [{'b': {'inner': 123}, 'c': 456}, {'e': 789}]
}
```

*   Extract values from a nested tree:

    ```python
    assert kontext.get_by_path(tree, 'a[0].b.inner') == 123
    ```

*   Flatten a nested tree:

    ```python
    assert kontext.flatten_with_path(tree) == {
        'a[0].b.inner': 123,
        'a[0].c': 456,
        'a[1].e': 789,
    }
    ```

Note: `a.b` can match both `a['b']` (index) or `a.b` (attribute).

## Keys

A `Key` is a named path:

```
input: Key = 'a[0].b.inner'
```

Here, the `input` key is assigned to the path value `a[0].b.inner`.

Keys are defined in a `KeyedObject`. A `KeyedObject` can be any arbitrary
Python object annotated with special `: Key` annotations:


```python
@dataclasses.dataclass
class A:
  x: Key
  y: Optional[Key]
```

Here `A` defines 2 keys: `x` and `y`.

Those keys can then be resolved with:

* `kontext.get_keypaths`: Get the key values.

  ```python
  a = A(x='a[0].b.inner', y=None)

  assert kontext.get_keypaths(a) == {
      'x': 'a[0].b.inner',
      'y': None,
  }
  ```

* `kontext.resolve_from_keyed_obj`: Get the key values and apply them to get the
  corresponding values from the tree.

  ```python
  a = A(x='a[0].b.inner', y='a[1].e')

  assert kontext.resolve_from_keyed_obj(tree, a) == {
      'x': 123,
      'y': 789,
  }
  ```

### [Advanced] Dynamically extract keys

Rather than hardcoding the available keys as annotations (`x: Key`), it is
possible to dynamically define keys through the `__kontext_keys__` protocol.

This can be used to propagate keys from an inner objects:

```python
@dataclasses.dataclass
class B:
  inner_keyed_obj: A

  def __kontext_keys__(self) -> dict[str, str | None]:
    return kontext.get_keypaths(self.inner_keyed_obj)


b = B(inner_keyed_obj=A(x='a[0].b.inner', y='a[1].e'))


assert kontext.get_keypaths(b) == {
    'x': 'a[0].b.inner',
    'y': None,
}
```

## Use case

In Kauldron, paths and keys are used to link batch, model, losses, metrics,...
together without having to hardcode any assumption on the batch structure,
model inputs,...:

Each model can define through Key what are the expected model inputs:

```python
class MyModel(nn.Module):
  img: kontext.Key = kontext.REQUIRED  # Match `__call__` signature
  label: kontext.Key = kontext.REQUIRED

  @nn.compact
  def __call__(self, img, label):
    ...
```

Then each user can specify in their config how the model inputs are mapped
to their specific batch:

```python
model = MyModel(
  img='batch.image',
  label='batch.label',
)
```

Kauldron internally uses `kontext` to extract the values from `batch` and
forward them to the `model`.

```python
context = {
    'batch': batch,
    ...
}
model_kwargs = kontext.resolve_from_keyed_obj(context, model)
pred = model.apply(rng, **model_kwargs)
```

### Helper

Rather than using string which can be fragile, it is possible to use
`kontext.path_builder_from` to dynamically generate the paths with auto-complete
and type checking.

Let's imaging your dataset yield some structured object (e.g.
`typing.TypedDict`, `dataclass`,...):

```python
@flax.struct.dataclass
class Batch:
  image: jnp.array
  label: jnp.array
```

In your config, you can replace the keys `str` by their typed version:

```python
batch = kontext.path_builder_from('batch', Batch)

model = MyModel(
  img=batch.image,  # < Auto-complete and attribute checking, rather than `str`
  label=batch.label,
)
```