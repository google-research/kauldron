# PyGrain data pipeline



[TOC]

`kd.data.py` is a small wrapper around `kd.data.Pipeline` to build flexible data
pipelines based on `PyGrain`.

## Example

Minimal example:

```python
cfg.train_ds = kd.data.py.Tfds(
    # TFDS parameters
    name='mnist',
    split='train',

    # `kd.data.py.PyGrainPipeline` optional parameters (common to all objects)
    batch_size=32,
    transforms=[
        kd.data.py.Elements(keep=["image"]),
    ],
)
```

Example of dataset mixture with nested transforms:

```python
cfg.train_ds = kd.data.py.Mix(
    datasets=[
        kd.data.Tfds(
            name='cifar100',
            split='train',
            transforms=[
                kd.data.py.Elements(keep=["image", "label"]),
            ],
        ),
        kd.data.Tfds(
            name='imagenet2012',
            split='train',
            transforms=[
                kd.data.py.Elements(keep=["image", "label"]),
                kd.data.py.Resize(key='image', height=32, width=32),
            ],
        ),
    ],
    seed=0,
    batch_size=256,
    transforms=[
        kd.data.py.RandomCrop(shape=(15, 15, None)),
    ],
)
```

## API

`pymix` provides the following sources:

* `kd.data.py.Tfds`: TFDS dataset (note this require the dataset to be in a
  format supporting random access, like ArrayRecord)
* `kd.data.py.DataSource`: Support arbitrary `grain.RandomAccessDataSource`.

Additionally, sources datasets can be combined using:

* `kd.data.py.Mix`: Sample from a combination of datasets.

## Implement your own

By default, any `grain.RandomAccessDataSource` can be used without subclassing
using `kd.data.py.DataSource(my_source)`. However for convenience,
it can be useful to create a small wrapper to remove nesting and expose your
dataset directly. In this case, you can inherit from `DataSourceBase` and
implement the `data_source` property:

```python
class Tfds(kd.data.py.DataSourceBase):
  name: str
  split: str

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    return tfds.data_source(self.name, split=self.split)
```
