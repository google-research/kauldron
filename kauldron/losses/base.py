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

"""Base class for defining losses."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, ClassVar, Literal, Optional

import flax
import jax
from jax import numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.metrics import base_state
from kauldron.typing import Array, Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member


Schedule = Callable[[int], float]


@flax.struct.dataclass
class AllReduceMean(base_state.State):
  """Default state for aggregating losses (tracks a scalar mean value)."""

  value: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def empty(cls) -> AllReduceMean:
    return cls(value=jnp.array(0, jnp.float32), count=jnp.array(0, jnp.float32))

  @classmethod
  def from_values(
      cls,
      values: jnp.ndarray,
      mask: jnp.ndarray | None = None,
      weight: float = 1.0,
      normalize_by: Literal["mask", "values"] = "mask",
  ) -> AllReduceMean:
    if mask is None:
      value = jnp.sum(values)
      count = jnp.array(values.size, dtype=jnp.float32)
    else:
      try:
        mask = jnp.broadcast_to(mask, values.shape)
      except Exception as e:
        raise ValueError(
            f"Mask {mask.shape} must always be broadcastable to value"
            f" {values.shape}."
        ) from e
      value = jnp.sum(values * mask)
      if normalize_by == "mask":
        count = jnp.sum(mask, dtype=jnp.float32)
      else:
        count = jnp.array(values.size, dtype=jnp.float32)
    return cls(value=value * weight, count=count)

  def merge(self, other: AllReduceMean) -> AllReduceMean:
    return type(self)(
        value=self.value + other.value,
        count=self.count + other.count,
    )

  def compute(self) -> Float[""]:
    return self.value / jnp.clip(self.count, min=1e-8)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Loss(metrics.Metric, abc.ABC):
  """Base class for losses which handles masks, averaging, and loss-weight.

  Subclasses should implement `get_values` which should compute the loss value
  and return `self.State.from_values(values=values, mask=mask)`

  Example:
    ```
    # Instantiate Loss with parameters and required keys:
    loss = SoftmaxCrossEntropy(logits="preds.logits", labels="batch.labels")

    # Shorthand computation given a context object ctx which contains
    # the logits and labels in the previously specified paths:
    value = loss(context=ctx)

    value = loss(logits=..., labels=...)  # directly passing logits and labels

    # The above shorthand is only recommended for interactive usage.
    # In training code use get_state and compute:
    loss_state = loss.get_state_from_context(ctx)        # from context
    loss_state = loss.get_state(logits=..., labels=...)  # directly

    value = loss_state.compute()
    ```

  Attributes:
    step: The key for determining the current step (for weight schedules).
    weight: Determines the weight of this loss term for the total loss. Can be
      either a float constant or a schedule (a function from step-number to
      float). Defaults to 1.0.
    mask: Optional key for a mask of values in [0, 1] which can be used to
      ignore parts of the batch. The shape of the mask should be broadcastable
      to the output shape of the `compute` function. Defaults to None.  Losses
      can be computed in two ways: 1. directly by passing the required arguments
      to the __call__ method. 2. using `apply_to_context` to automatically
      gather the arguments from a given context. This takes into account the
      weight of the loss.
    normalize_by: Whether to divide the total loss over the number of mask
      elements (normalize_by = "mask"), or over the total number of values
      (normalize_by = "values"). Defaults to "mask".
  """

  step: kontext.Key = "step"
  mask: Optional[kontext.Key] = None
  weight: int | float | Schedule = 1.0
  normalize_by: Literal["mask", "values"] = "mask"

  State: ClassVar[type[AllReduceMean]] = (  # pylint: disable=invalid-name
      AllReduceMean
  )

  @abc.abstractmethod
  def get_values(self, *args, **kwargs) -> Array["..."]:
    """Compute the loss values (before masking, averaging and weighting).

    Subclasses need to implement this method.
    Args:
      *args: Any required arguments (names should match kontext.Key annotations)
      **kwargs: Any arguments (names should match kontext.Key annotations)

    Returns:
      A jnp.Array of loss values compatible in shape with any desired masking.
    """
    ...

  def get_state(
      self,
      *args,
      mask: Optional[Array["..."]] = None,
      step: Optional[int] = None,
      **kwargs,
  ) -> Loss.State:
    """Compute the loss state, and takes care of masking and loss-weight.

    The Loss.State is AllReduceMean by default which keeps track of a single
    scalar loss value, but ensures correctly averaging even while using masks.

    Args:
      *args: Positional arguments to be passed on to `get_values`.
      mask: An optional mask to exclude some of the loss values from the total.
        The shape of this mask needs to be broadcastable to the shape of values
        returned from `get_values`. A value of 1 means that a value should be
        included (and 0 to exclude).
      step: The current step to be used to compute the loss-weight if
        `self.weight` is set to a schedule. Otherwise `step` is ignored.
      **kwargs: Keyword arguments to be passed on to `get_values`.

    Returns:
      An instance of Loss.State (AllReduceMean by default) which keeps track of
      a single scalar loss value, but ensures correctly averaging even while
      using masks. This final loss value can be computed from this state by
      calling state.compute().
      Optionally the state first can be reduced (to remove the device dimension
      after pmap) or merged with other (previous) loss states.
    """

    values = self.get_values(*args, **kwargs)
    weight = self.get_weight(step=step)
    return self.State.from_values(
        values=values, mask=mask, weight=weight, normalize_by=self.normalize_by
    )

  def get_state_from_context(self, context: Any) -> Loss.State:
    """Compute the loss-state by auto-filling args from given context.

    This is a wrapper around `get_state` that gathers the required arguments
    from the given context, using `kontext.Key`s of the loss.
    For example if the loss has a `target : kontext.Key` set to `"batch.label"`,
    then `context.batch["label"]` will be passed to the `get_state` function
    of the loss.

    Args:
      context: A context object that holds the information (e.g. the current
        batch and the model outputs) against which the kontext.Keys of the loss
        are resolved.

    Returns:
      An instance of Loss.State. See `get_state` for details.
    """
    # TODO(epot): Add `func=self.get_values`
    kwargs = kontext.resolve_from_keyed_obj(context, self)
    mask = kwargs.pop("mask", None)
    step = kwargs.pop("step", None)
    values = self.get_values(**kwargs)
    weight = self.get_weight(step=step)
    return self.State.from_values(
        values=values, mask=mask, weight=weight, normalize_by=self.normalize_by
    )

  def get_weight(self, step: Optional[int] = None) -> Float[""]:
    """Return the weight of this loss at the given step number.

    Args:
      step: If the loss is set to a schedule, then this is the step used for
        computing the weight. Otherwise it is unused/optional.

    Returns:
      The weight of this loss term for the total loss (for the given step).
    """
    if isinstance(self.weight, (float, int)):
      return jnp.array(self.weight, dtype=jnp.float32)
    else:
      if step is None:
        raise ValueError("Weight is a schedule, so step is required.")
      return self.weight(step)

  def __call__(
      self,
      *,
      context: Optional[Any] = None,
      **kwargs,
  ) -> Float[""]:
    """Shorthand to evaluate the loss either from context or kwargs.

    Is equivalent to first calling `get_state` (or `get_state_from_context`) and
    then `compute`. Mostly meant as a convenient interface for interactive use.

    Args:
      context: Any pytree that can be used as context to resolve the required
        kwargs of this loss. Cannot be used together with explicit **kwargs.
      **kwargs: The arguments for `get_state`. Cannot be used together with
        `context`.

    Returns:
      Scalar value of the loss.
    """
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or explicit kwargs, but got context and"
            f" {kwargs.keys()}."
        )
      state = self.get_state_from_context(context)
    else:
      state = self.get_state(**kwargs)
    return state.compute()


@jax.named_call
def compute_losses(
    losses: PyTree[Loss], context: kontext.Context
) -> tuple[Float[""], PyTree[Float[""]]]:
  """Compute all losses based on given context."""

  loss_states = jax.tree.map(
      lambda loss: loss.get_state_from_context(context), losses
  )
  loss_values = jax.tree.map(
      lambda state: state.compute(),
      loss_states,
      is_leaf=base_state.State.isinstance,
  )

  total_loss = jax.tree.reduce(
      jnp.add,
      loss_values,
      initializer=jnp.asarray(0.0),
  )  # pytype: disable=wrong-arg-types  # numpy-scalars
  return total_loss, loss_states
