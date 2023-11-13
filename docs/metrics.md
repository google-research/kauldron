# Metrics

Loss, metrics, summaries all share the same API.

See the available metrics:

*   Metrics: https://github.com/google-research/kauldron/tree/HEAD/kauldron/metrics/__init__.py
*   Losses: https://github.com/google-research/kauldron/tree/HEAD/kauldron/losses/__init__.py
*   Summaries: https://github.com/google-research/kauldron/tree/HEAD/kauldron/summaries/__init__.py

## Using a metric

### Standalone usage

Metrics can be used outside Kauldron, as standalone module.

Metrics are stateless objects.

Creation:

```python
metric1 = kd.metric.Accuracy()
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

## Creating a metric

Metrics inherit the `kd.metrics.Metric` class and overwrite the `State` class
and `get_state` attribute.

```python
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Accuracy(kd.metrics.Metric):
  """Classification Accuracy."""

  logits: kontext.Key = "preds.logits"
  labels: kontext.Key = "batch.label"

  # Could be `State = kd.metrics.AverageState` but inheritance give a better
  # name `Accuracy.State`
  class State(kd.metrics.AverageState):
    pass

  @typechecked
  def get_state(self, logits: Float["*b n"], labels: Float["*b"]) -> Float["*b"]:
    correct = logits.argmax(axis=-1) == labels
    return self.State.from_values(values=correct)
```

2 states are provided by default:

*   `kd.metrics.AverageState`: for simple averaging of a value (e.g.
    `kd.metrics.Norm`).
*   `kd.metrics.CollectingState`: for metrics that need to collect and
    concatenate model outputs over many batches (e.g. `kd.metrics.RocAuc`).
