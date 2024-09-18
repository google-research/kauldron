# tf.data pipelines

https://kauldron.rtfd.io/en/latest-kmix

[TOC]

`kd.data.tf` is a small wrapper around `kd.data.Pipeline` to build flexible data
pipelines based on `tf.data`.

## Example

Minimal example:

```python
cfg.train_ds = kd.data.tf.Tfds(
    # TFDS parameters
    name='mnist',
    split='train',

    # `kd.data.tf.TFDataPipeline` optional parameters (common to all objects)
    batch_size=32,
    transforms=[
        kd.data.Elements(keep=["image"]),
    ],
)
```

Example of dataset mixture with nested transforms:

```python
cfg.train_ds = kd.data.tf.SampleFromDatasets(
    datasets=[
        kd.data.tf.Tfds(
            name='cifar100',
            split='train',
            transforms=[
                kd.data.Elements(keep=["image", "label"]),
            ],
        ),
        kd.data.tf.Tfds(
            name='imagenet2012',
            split='train',
            transforms=[
                kd.data.Elements(keep=["image", "label"]),
                kd.data.Resize(key='image', height=32, width=32),
            ],
        ),
    ],
    seed=0,
    batch_size=256,
    transforms=[
        kd.data.RandomCrop(shape=(15, 15, None)),
    ],
)
```

## API

`kd.data.tf` provides the following sources:

* `kd.data.tf.Tfds`: TFDS dataset (note this require the dataset to be
  ArrayRecord format)
* `kd.data.tf.TfdsLegacy`: TFDS dataset for datasets not supporting random
  access (e.g. in `tfrecord` format)
* `kd.data.tf.SeqIOTask`: SeqIO task
* `kd.data.tf.SeqIOMixture`: SeqIO mixture

Other sources will be added in the future. If your dataset is not yet supported,
please [contact us](https://kauldron.rtfd.io/en/latest-help#bugs-feedback).

<!--

TODO(epot): Add more source options.

-->

Additionally, sources datasets can be combined using:

* `kd.data.tf.SampleFromDatasets`: Sample from a combination of datasets.

## Implement your own

All classes inherit from this simple protocol.

```python
class TFDataPipeline(kd.data.Pipeline):

  @abc.abstractmethod
  def ds_for_current_process(self) -> tf.data.Dataset:
    ...
```

### Implementing a source dataset

> Important: Sources should take care of:
>
> * **Sharding**: Each host should yield non-overlapping examples.
> * **Deterministic shuffling** (usually with a `shuffle: bool` kwargs). Ideally using
>   random-access shuffling (provided by TFGrain, ArrayRecord,...).
> * **`num_epoch`**: Repeating the dataset for a given number of epochs (while
  making sure each epoch reshuffle the data).

`transforms`, `batch_size`,... are automatically taken cared of.

See `kd.data.tf.Tfds` for an example.

If your dataset does not support random access, you can inherit from
`kd.data.tf.WithShuffleBuffer` that will automatically take care of adding
`ds.cache`, `ds.shuffle`, `ds.repeat`. Note that the source dataset should
still make sure each process yields non-overlapping examples.

Transformations should be added using `transforms=` kwargs, rather than
hardcoded in the implementation.

<code class="lang-python"><pre>def ds_for_current_process(self, rng: kd.random.PRNGKey) -> tf.data.Dataset:
  ...<del>
  ds = _common_transform(ds)
  ds = ds.map(_resize_image)</del>
  return ds
</pre></code>

<code class="lang-python"><pre>cfg.train_ds = MyDataset(
    transforms=[<ins>
        MyDatasetTransform(),
        kd.data.Resize(key='image', height=32, width=32),</ins>
    ],
)
</pre></code>

Please comment on b/284152266 if you need your transformation to be added
before the shuffle buffer so this get prioritized.

### Implementing a mixture transformation

Mixtures should take care of:

* **Splitting the rng** on each of the sub-dataset. To ensure sub-dataset yields
  examples in different order.
* **Calling `sub_ds.ds_with_transforms()`** to access the sub-datasets. You
should **not** call `sub_ds.ds_for_current_process()` directly as this would
skip the transformations from the sub dataset.

See `kd.data.tf.SampleFromDatasets` for an example.

## Pipeline API

Like all `kd.data.Pipeline` objects, the dataset can be used directly as a
standalone iterator.

```python
ds = kd.data.Tfds(...)
ds = ds.device_put(kd.sharding.FIRST_DIM)

for ex in ds:
  ...
```
