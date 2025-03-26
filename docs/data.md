# Data pipelines

## Pipelines options

All Kauldron data pipelines inherit from the `kd.data.Pipeline` abstract base
class. By default, Kauldron provides two main pipelines implementations:

*   Recommended: PyGrain based: `kd.data.py`, which takes any custom
    `grain.RandomAccessDataSource` as input. Example:

    ```python
    cfg.train_ds = kd.data.py.DataSource(
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

## Transformations

Both `kd.data.py.PyGrainPipeline` and `kd.data.tf.TFDataPipeline` can be
customized through a list of grain transformations.

Common transforms include:

*   `kd.data.Elements`: Filter which fields to keep/drop in the batch
*   `kd.data.ValueRange`: Normalize a tensor (e.g. uint8 image to `-1, 1`)
*   `kd.data.Rearrange`: Reshape a tensor (using `einops` notation)
*   `kd.data.tf.Resize`: Resize an image
*   See `kd.data` for a full list.

`kd.contrib.data` also contain a lot of additional transformations.

Note: Most transformation in Kauldron only supports `kd.data.tf.TFDataPipeline`.
Please let us know if you need PyGrain support for any specific transformation.
