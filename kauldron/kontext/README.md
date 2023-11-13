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
@dataclass
class A:
  x: Key
  y: Optional[Key]
```

Here `A` defines 2 keys: `x` and `y`.

Those keys can then be resolved with `get_from_keys_obj`:

```python
a = A(x='a[0].b.inner', y='a[1].e')

assert kontext.get_from_keys_obj(tree, a) == {
    'x': 123,
    'y': 789,
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
forward them to the `model``.

```python
context = {
    'batch': batch,
    ...
}
model_kwargs = kontext.get_from_keys_obj(context, model)
pred = model.apply(rng, **model_kwargs)
```
