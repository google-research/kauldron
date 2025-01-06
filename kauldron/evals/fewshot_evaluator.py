# Copyright 2024 The kauldron Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluators."""

from __future__ import annotations

import dataclasses
import functools
from typing import Mapping, Sequence, TypeVar

import flax
import flax.struct
import jax
from jax import numpy as jnp
from kauldron import data
from kauldron import kontext
from kauldron.evals import evaluators
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.train import train_step
from kauldron.typing import Array, Float, Int, Scalar, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import config_util
from kauldron.utils import kdash
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
import numpy as np

_SelfT = TypeVar('_SelfT')


class FewShotEvaluator(evaluators.EvaluatorBase):
  """FewShotEvaluator running closed-form few-shot classification.

  Compute the features from the model, solve closed-form L2-regularized linear
  regression for few-shot classification. This is fairly fast, so can be run
  regularly during training.

  Following (and largely copying) https://github.com/google-research/big_vision

  Attributes:
    ds_train: Dataset to train few-shot classification on
    ds_val: Dataset to validate few-shot classification on (to select L2 reg)
    ds_test: Dataset to test few-shot classification on
    metric_prefix: String prefix to be used for the metrics from this evaluator
    num_classes: Number of classes in the classification task
    num_shots: A sequence of integers - numbers of shots to be evaluated
    repr_names: A dictionary of representations to be evaluated. Keys are names
      to be used to refer to the representations, values are paths in the
      context from which to take the actual features
    l2_regs: Possible values for L2 regularization.
    label_name: key by which to get the labels from the context
    selected_repr: a key from repr_names for which to put the accuracies to the
      main metrics
    seed: random seed for selecting the training data subset

  Usage example:
    "fewshot_i1k": kd.evals.FewShotEvaluator(
        run=kd.evals.EveryNSteps(10_000),
        metric_prefix="i1k",
        ds_train=_make_i1k_fewshot(split="train[:-10000]", batch_size=4096),
        ds_val=_make_i1k_fewshot(split="train[-10000:]", batch_size=4096),
        ds_test=_make_i1k_fewshot(split="validation", batch_size=4096),
        num_classes=1000,
        num_shots=(1, 2, 5, 10),
        repr_names={"pre_logits": "interms.pre_logits.__call__[0]"},
        label_name="batch.label",
    )
  """

  ds_train: data.Pipeline
  ds_val: data.Pipeline
  ds_test: data.Pipeline
  metric_prefix: str
  num_classes: int
  num_shots: Sequence[int]
  repr_names: Mapping[str, str] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  l2_regs: Sequence[float] = (2**6, 2**7, 2**8, 2**9, 2**10)
  label_name: str
  selected_repr: str = 'pre_logits'
  seed: int | Sequence[int] = config_util.ROOT_CFG_REF.seed

  @property
  def seeds(self) -> list[int]:
    return list(self.seed) if isinstance(self.seed, Sequence) else [self.seed]

  def evaluate(self, state: train_step.TrainState, step: int):
    """Run one full evaluation."""
    self._assert_root_cfg_resolved()

    train_features, train_labels = self.compute_features(
        state, self.ds_train, 'train'
    )
    val_features, val_labels = self.compute_features(state, self.ds_val, 'val')
    test_features, test_labels = self.compute_features(
        state, self.ds_test, 'test'
    )

    fewshot_accuracies = {}
    for feat_key in train_features.keys():

      results = np.array([
          run_fewshot(
              train_features[feat_key],
              train_labels,
              val_features[feat_key],
              val_labels,
              test_features[feat_key],
              test_labels,
              num_classes=self.num_classes,
              all_shots=self.num_shots,
              l2_regs=self.l2_regs,
              seed=seed,
          )
          for seed in self.seeds
      ])
      check_type(results, Float['seeds 2 shots regs'])
      results_val = results[:, 0].mean(axis=0)
      results_test = results[:, 1].mean(axis=0)
      check_type(results_val, Float['shots regs'])
      check_type(results_test, Float['shots regs'])

      for shot_idx, shots in enumerate(self.num_shots):
        best_reg_idx = np.argmax(results_val[shot_idx])
        if feat_key == self.selected_repr:
          fewshot_accuracies[f'metrics/{self.metric_prefix}-{shots}shot'] = (
              results_test[shot_idx, best_reg_idx]
          )
        fewshot_accuracies[
            f'z_fewshot_all/{self.metric_prefix}-{feat_key}-{shots}shot'
        ] = results_test[shot_idx, best_reg_idx]
        for reg_idx, l2_reg in enumerate(self.l2_regs):
          l2_reg = float(l2_reg)
          fewshot_accuracies[
              f'z_fewshot_all/z_{self.metric_prefix}-{feat_key}-{shots}shot-{l2_reg:.5f}'
          ] = results_test[shot_idx, reg_idx]

    with jax.transfer_guard('allow'):
      self.writer.write_scalars(
          step=step,
          scalars=fewshot_accuracies,
      )
      self.writer.flush()
    return None

  def compute_features(
      self,
      state: train_step.TrainState,
      ds: data.IterableDataset,
      split: str,
  ) -> tuple[dict[str, Array['...']], Array['...']]:
    merged_aux = None
    for eval_step, batch in utils.enum_iter(
        ds.device_put(self.base_cfg.sharding.ds),
        desc=f'{self.name}_{split}',
    ):
      eval_step = sharding.device_put(eval_step, sharding.REPLICATED)
      aux = evaluators.basic_eval_step(  # pylint: disable=protected-access
          model_with_aux=self.model_with_aux,
          rng_streams=self.base_cfg.rng_streams,
          eval_step=eval_step,
          state=state,
          batch=batch,
          sharding=self.base_cfg.sharding,
      )
      # By default, cross-process communication is only allowed inside
      # `jax.jit` but clu metric do not support `jax.jit`:
      # https://github.com/google/CommonLoopUtils/tree/HEAD/clu/metrics.py;l=383;rcl=559340497
      # So we locally allow cross-process communication for merging the
      # metrics
      with jax.spmd_mode('allow_all'), jax.transfer_guard('allow'):
        merged_aux = merged_aux | aux
    assert merged_aux is not None  # At least one iteration
    merged_summaries = merged_aux.compute()
    features = {
        k.removeprefix('metrics/'): v
        for k, v in merged_summaries.metric_values.items()
    }
    labels = features.pop('labels')
    return features, labels

  @functools.cached_property
  def model_with_aux(self) -> train_step.ModelWithAux:
    """Model which also compute the auxiliaries (losses, metrics,...)."""
    return train_step.ModelWithAux(
        model=self.base_cfg.model,
        metrics=flax.core.FrozenDict(
            {
                key: ComputeFeaturesMetric(features=feature)
                for key, feature in self.repr_names.items()
            }
            | {'labels': ComputeFeaturesMetric(features=self.label_name)}
        ),
        losses=flax.core.FrozenDict({}),
        summaries=flax.core.FrozenDict({}),
    )

  @functools.cached_property
  def __dashboards__(self) -> kdash.DashboardsBase:
    return kdash.MetricDashboards(
        collection=self.name,
        metrics={
            f'{self.metric_prefix}-{shots}shot': None
            for shots in self.num_shots
        },
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ComputeFeaturesMetric(base.Metric):
  """Compute the features over a dataset."""

  features: kontext.Key

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    features: Array['...']

    @typechecked
    def compute(self) -> Array['...']:
      return np.array(super().compute().features)

  @typechecked
  def get_state(
      self,
      features: Array['...'],
  ) -> ComputeFeaturesMetric.State:
    # simply collect the given values
    return self.State(features=features)


BIAS_CONSTANT = 100.0


def to_cpu(x: Array['any*']) -> Array['any*']:
  return jax.device_put(x, jax.local_devices(backend='cpu')[0])


@typechecked
def run_fewshot(
    x_train_all: Float['n_tr d'],
    y_train_all: Int['n_tr'],
    x_val: Float['n_v d'],
    y_val: Int['n_v'],
    x_test: Float['n_t d'],
    y_test: Int['n_t'],
    num_classes: int,
    all_shots: tuple[int, ...],
    l2_regs: tuple[float, ...],
    seed: int = 17,
) -> tuple[Float['shots regs'], Float['shots regs']]:
  """Run few-shot evaluation."""
  rng = np.random.default_rng(seed)

  class_indices = [
      rng.permutation(np.where(y_train_all == cls_i)[0])
      for cls_i in range(num_classes)
  ]

  results_val = np.zeros((len(all_shots), len(l2_regs)))
  results_test = np.zeros((len(all_shots), len(l2_regs)))
  for shot_idx, shots in enumerate(all_shots):
    all_idx = [indices[:shots] for indices in class_indices]
    all_idx = np.concatenate(all_idx, axis=0)
    assert len(all_idx) == num_classes * shots, (
        f'expected {num_classes * shots} training samples for'
        f' {num_classes} classes and {shots} shots, instead got {len(all_idx)}'
    )
    x = x_train_all[all_idx]
    y = y_train_all[all_idx]

    # print(f'[fewshot][i1k][{shots}-shot]: compute cache')
    cache = _precompute_cache(to_cpu(x), to_cpu(y), num_classes)
    for l2_reg_idx, l2_reg in enumerate(l2_regs):
      acc_val = _eig_fewshot_acc_fn(
          cache, to_cpu(x_val), to_cpu(y_val), to_cpu(l2_reg)
      )
      results_val[shot_idx, l2_reg_idx] = acc_val
      acc_test = _eig_fewshot_acc_fn(
          cache, to_cpu(x_test), to_cpu(y_test), to_cpu(l2_reg)
      )
      results_test[shot_idx, l2_reg_idx] = acc_test
  return results_val, results_test


# The below functions are adapted from
# https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/fewshot_lsr.py


# Setup function for few-shot regression on CPU to avoid "polluting" the TPU.
@functools.partial(jax.jit, backend='cpu', static_argnums=(2,))
@typechecked
def _precompute_cache(
    x: Float['n d'],
    y: Int['n'],
    num_classes: int,
) -> _FewShotCache:
  """Cache quantities to speed-up the computation of L2-regularized least-sq."""
  # Whiten
  mean = jnp.mean(x, axis=0, keepdims=True)
  std = jnp.std(x, axis=0, keepdims=True) + 1e-5
  x = (x - mean) / std

  # Add a constant feature for the bias, large so it's almost unregularized:
  x = jnp.pad(x, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  # To one-hot representation rescaled into {-1, 1}
  y = 2.0 * jax.nn.one_hot(y, num_classes) - 1.0

  num_points, dim = x.shape
  # Let N be the number of points, D the dimension and C the number of classes.
  # We have x of shape (N, D) and y of shape (N, C).
  # For least-squares, we can compute
  #
  #   (A) when N >= D, (x^T x + l2 Id)^{-1} x^T y
  #   (B) when D > N, x^T  (x x^T + l2 Id)^{-1} y
  #
  # We pre-compute the eigen-decomposition of either x^T x or x x^T which
  # becomes q diag(eigs) q^T with q unitary matrix either (D, D) or (N, N)
  # and eigs a vector (D,) or (N,).
  #
  # For any l2 > 0, we can compute (x^T x + l2 Id)^{-1} or (x x^T + l2 Id)^{-1}
  # by simply computing q (diag(eigs) + l2 Id)^{-1} q^T.
  # (SVD would be more natural here, but it proved slower, so we use eigh)
  #
  # Both cases (A) and (B) can be viewed as lhs (diag(eigs) + l2 Id)^{-1} rhs,
  # where lhs/rhs are pre-computed left/right-hand sides to specify.
  #
  if num_points >= dim:
    eigs, q = jnp.linalg.eigh(x.T @ x)
    rhs = q.T @ (x.T @ y)
    lhs = q
  else:
    eigs, q = jnp.linalg.eigh(x @ x.T)
    rhs = q.T @ y
    lhs = x.T @ q

  return _FewShotCache(eigs=eigs, rhs=rhs, lhs=lhs, mean=mean, std=std)


@flax.struct.dataclass
class _FewShotCache:
  eigs: Float['d'] | Float['n']
  rhs: Float['d d'] | Float['n n']
  lhs: Float['n n'] | Float['d d']
  mean: Float['1 d']
  std: Float['1 d']


@functools.partial(jax.jit, backend='cpu')
@typechecked
def _eig_fewshot_acc_fn(
    cache: _FewShotCache,
    x_test: Float['m d'],
    y_test: Int['m'],
    l2_reg: Scalar,
) -> Scalar:
  """Computes (x,y) linear regression accuracy on (x_test, y_test)."""

  x_test = (x_test - cache.mean) / cache.std
  x_test = jnp.pad(x_test, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  # See comments in _precompute_cache for context about the formula.
  scaling = 1.0 / (cache.eigs + l2_reg * jnp.ones_like(cache.eigs))
  scaling = scaling.reshape((1, -1))
  w = (cache.lhs * scaling) @ cache.rhs
  # Predict test-set values and measure their accuracy
  preds = jnp.argmax(x_test @ w, axis=1)
  return jnp.mean(preds == y_test)
