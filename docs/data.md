# Train, eval, randomness

## Pipelines options

All Kauldron data pipelines inherit from the `kd.data.Pipeline` abstract base class.
By default, Kauldron provides two main pipelines implementations:

*   Recommended: `tf.data` based: `kd.data.TFDataPipeline` base class which
    itself implements multiple sub-classes (see next section). For example:

    ```python
    cfg.train_ds = kd.data.Tfds(
        # TFDS specific args
        name='mnist',
        split='train',
        shuffle=True,

        # `kd.data.TFDataPipeline` args (common to all TFDataPipeline)
        batch_size=256,
        transforms=[
            kd.data.Elements(keep=["image"]),
            kd.data.ValueRange(key="image", vrange=(0, 1)),
        ],
    )
    ```

*   PyGrain based: `kd.data.PyGrainPipeline`, which takes any custom
    `grain.RandomAccessDataSource` as input. Example:

    ```python
    cfg.train_ds = kd.data.PyGrainPipeline(
        data_source=tfds.data_source("mnist", split='train'),
        shuffle=True,
        batch_size=256,
        transforms=[
            kd.data.Elements(keep=["image"]),
            kd.data.ValueRange(key="image", vrange=(0, 1)),
        ],
    )
    ```

While it's easy to implement your custom pipeline, please contact us if the
existing pipelines do not fit your use-case.

## TFDataPipeline

The following `tf.data` sources are available:

*   `kd.data.Tfds`: TFDS dataset (note that this requires the dataset to be in
    ArrayRecord format)
*   `kd.data.TfdsLegacy`: TFDS dataset for datasets not supporting random access
    ( e.g. in `tfrecord` format)
*   `kd.data.SeqIOTask`: SeqIO task
*   `kd.data.SeqIOMixture`: SeqIO mixture

Additionally, any of those sources dataset can be combined using:

*   `kd.data.SampleFromDatasets`: Sample from a combination of datasets.

Other sources will be added in the future. If your dataset is not yet supported,
please [contact us](https://kauldron.rtfd.io/en/latest-help#bugs-feedback).

See https://kauldron.rtfd.io/en/latest-kmix for details on how to implement a custom `tf.data` source.

Example of dataset mixture with nested transforms:

```python
cfg.train_ds = kd.kmix.SampleFromDatasets(
    datasets=[
        kd.kmix.Tfds(
            name='cifar100',
            split='train',
            transforms=[
                kd.data.Elements(keep=["image", "label"]),
            ],
        ),
        kd.kmix.Tfds(
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

## Transformations

Both `kd.data.PyGrainPipeline` and `kd.data.TFDataPipeline` can be customized
through a list of grain transformations.

Common transforms include:

*   `kd.data.Elements`: Filter which fields to keep/drop in the batch
*   `kd.data.ValueRange`: Normalize a tensor (e.g. uint8 image to `-1, 1`)
*   `kd.data.Rearrange`: Reshape a tensor (using `einops` notation)
*   `kd.data.Resize`: Resize an image
*   See `kd.data` for a full list.

`kd.contrib.data` also contain a lot of additional transformations.

Note: Most transformation in Kauldron only supports `kd.data.TFDataPipeline`.
Please let us know if you need PyGrain support for any specific transformation.
