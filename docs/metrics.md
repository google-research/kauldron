# Metrics

Loss, metrics, summaries all share the same API.

See the available metrics:

*   [Metrics](https://github.com/google-research/kauldron/tree/main/kauldron/metrics/__init__.py)
*   [Losses](https://github.com/google-research/kauldron/tree/main/kauldron/losses/__init__.py)
*   [Summaries](https://github.com/google-research/kauldron/tree/main/kauldron/summaries/__init__.py)

## Using a metric

### Kauldron usage

In Kauldron, the metrics are automatically applied and accumulated by the
training loop. User specify what are the metrics inputs through the
[keys](https://github.com/google-research/kauldron/blob/main/docs/intro.md?cl=head#keys-and-context).

```python
cfg.metrics = {
    'reconstruction': kd.losses.L2(preds="preds.image", targets="batch.image"),
    'roc_auc': kd.metrics.RocAuc(preds="preds.logits", targets="batch.label"),
}
```

### Standalone usage

Metrics can be used outside Kauldron, as standalone module (using
`//third_party/py/kauldron/metrics`,...):

```python
from kauldron import metrics
from kauldron import losses
from kauldron import summaries
```

Metrics are stateless objects.

Creation:

```python
metric1 = metrics.Accuracy()
```

Usage (1-time):

```python
accuracy = metric(logits=logits, labels=labels)
```

Equivalent to:

```python
accuracy = metric.get_state(logits=logits, labels=labels).compute()
```

Usage (accumulated):

Some metrics require accumulating values over multiple steps. In this case,
every metrics can emit a states which are merged together:

```python
state0 = metric.get_state(logits=logits, labels=labels)
state1 = metric.get_state(logits=logits, labels=labels)

# Accumulate the states
acc_state = state0.merge(state1)

# Compute the final metric value
accuracy = acc_state.compute()
```

## Creating a metric

### Metric

Metrics inherit the `kd.metrics.Metric` class and overwrite the `State` class
and `get_state` attribute.

```python
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Accuracy(kd.metrics.Metric):
  """Classification Accuracy."""

  logits: kontext.Key = kd.kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kd.kontext.REQUIRED  # e.g. "batch.label"

  # Could be `State = kd.metrics.AverageState` but inheritance give a better
  # name `Accuracy.State`
  class State(kd.metrics.AverageState):
    pass

  @typechecked
  def get_state(self, logits: Float["*b n"], labels: Float["*b"]) -> Float["*b"]:
    correct = logits.argmax(axis=-1) == labels
    return self.State.from_values(values=correct)
```

The state perform the agregation of the metric values. Some states are provided
by default:

*   `kd.metrics.AverageState`: for simple averaging of a value (e.g.
    `kd.metrics.Norm`).
*   `kd.metrics.CollectingState` (deprecated): for metrics that need to collect
    and concatenate model outputs over many batches (e.g. `kd.metrics.RocAuc`).
*   `kd.metrics.AutoState`: A more flexible version of `CollectingState` which
    supports arbitrary aggregation. See `Summary` section below.

You can also implement your custom `State`. To choose whether the logic should
go in `State` or `Metric`:

*   `Metric.get_state`: Is executed inside `jax.jit`
*   `State.compute`: Is executed outside `jax.jit`, so can contain arbitrary
    Python code (e.g. some `sklearn.metrics` contains logic which would be hard
    to implement in pure Jax)

### Loss

All losses inherit from `kd.metrics.Metric`, so uses the same API as above.
However for convenience, a `kd.losses.Loss` base class is provided which also
supports handling masks, averaging, and loss-weight.

The difference is that `Loss` implement the `get_values` method instead of
`get_state` (and the `get_state` method is implemented by the base class).

```python
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class L2(kd.losses.Loss):
  """L2 loss."""

  preds: kontext.Key = kd.kontext.REQUIRED
  targets: kontext.Key = kd.kontext.REQUIRED

  @typechecked
  def get_values(self, preds: Float["*b"], targets: Float["*b"]) -> Float["*b"]:
    return jnp.square(preds - targets)
```

Note: The `get_values` method returns the per-example loss (i.e. the returned
value has the batch dimension), and the averaging is done in the base class.
This ensure accumulating losses over multiple batches (e.g. in eval) works
correctly.

Losses also adds:

*   `weight`: If multiple losses are used, they can be weighted differently.
*   `mask`: Allow to filter out examples from the batch.

### Summary

Summaries are like metrics. The only difference is the `State.compute()` method
do not returns a scalar but instead returns one of:

*   `Array['b h w c']`: for image summaries.
*   `str` for text summaries.
*   `kd.summaries.Histogram`: for histogram summaries.
*   `kd.summaries.PointCloud`: for point cloud summaries.

For convenience, the `State` can inherit from `kd.metrics.AutoState` to perform
automated aggregations (e.g. only keep the first `x` images):

```python

@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowImages(kd.metrics.Metric):
  num_images: int = 5

  @flax.struct.dataclass
  class State(kd.metrics.AutoState):
    """Collects the first num_images images and boxes."""

    # When the states are agregated (e.g. across multiple batches, only keep
    # the first `num_images` images).
    images: Float["n h w c"] = kd.metrics.truncate_field(num_field="parent.num_images")
    boxes: Float["n k 4"] = kd.metrics.truncate_field(num_field="parent.num_images")

    ...

  ...
```
