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

"""Defines the AutoState base class along with its field mergers."""

from __future__ import annotations

import abc
import dataclasses
import types
from typing import Any, Literal, Self, TypeAlias, TypeVar

import jax
from kauldron import kontext
from kauldron.metrics import base_state
from kauldron.metrics.base_state import EMPTY  # pylint: disable=g-importing-member
from kauldron.typing import Array  # pylint: disable=g-multiple-import
import numpy as np

_MetricT = TypeVar("_MetricT")
_SelfT = TypeVar("_SelfT")
Empty: TypeAlias = Literal[base_state._EMPTY_TYPE.EMPTY]  # pylint: disable=protected-access


class AutoState(base_state.State[_MetricT]):
  """Flexible base class for conveniently defining custom states.

  Subclasses of AutoState have to use the @flax.struct.dataclass decorator and
  can define two kinds of fields:

  1) Data fields are defined by the `sum_field`, `concat_field` or
     `truncate_field` functions. E.g. `d : Float['n'] = sum_field()`.
     Data fields are pytrees of Jax arrays.
     They are merged by summing, concatenating or truncating, respectively.

  2) All other fields are static fields which are not merged, and instead
     checked for equality during `merge`. They are also not pytree nodes, so
     they are not touched by jax transforms (but can lead to recompilation if
     changed). These can be useful to store some parameters of the metric,
     e.g. the number of elements to keep. Note that static fields are rarely
     needed, since it is usually better to define static params in the
     corresponding metric and access them through the `parent` field.

  The `compute` method by default returns a namespace with the final data field
  values (as np.ndarrays). It can be overridden to return any other object, but
  should still call `super().compute()` to finalize the data fields.

  Example:
  ```python
  @flax.struct.dataclass(kw_only=True)
  class CustomErrorSummaryState(kd.metrics.AutoState):
    # static-fields
    cmap: str = "coolwarm"
    num_to_keep: int = 5
    num_buckets: int = 30

    # data-fields
    error: Float['n h w 1'] = kd.metrics.truncate_field(num_field="num_to_keep")
    summed_error: Float[''] = kd.metrics.sum_field()
    total_error: Float[''] = kd.metrics.sum_field()
    error_hist: Float['n'] = kd.metrics.concat_field()

    def compute(self):
      # NOTE: You should access the data-fields through super().compute()
      # rather than through self, to ensure that they are properly finalized.
      # (e.g. converted to np.ndarrays)
      data = super().compute()
      error_img = mediapy.to_rgb(data.error, cmap=self.cmap)
      return {
          "avg_error": data.summed_error / data.total_error,
          "error_images": error_img,
          "error_hist": kd.summaries.Histogram(
              tensor=data.error_hist, num_buckets=self.num_buckets
          ),
      }
  ```
  """

  @classmethod
  def empty(cls: type[Self]) -> Self:
    # Set all fields to EMPTY to allow merging with the empty state.
    empty_fields = {f.name: EMPTY for f in dataclasses.fields(cls)}
    return cls(**empty_fields)

  def merge(self: _SelfT, other: _SelfT) -> _SelfT:
    """Checks static fields for equality and merges data-fields."""
    # first check static fields for equality and resolve any EMPTY fields
    # this is needed for truncate to work when merging with empty states
    static_fields = {}
    for field in dataclasses.fields(self):
      if _is_static_field(field):
        static_fields[field.name] = _assert_static_field_equal(
            getattr(self, field.name), getattr(other, field.name), field
        )
    updated_self = dataclasses.replace(self, **static_fields)

    merged_fields = {}
    for field in dataclasses.fields(self):
      v1 = getattr(self, field.name)
      v2 = getattr(other, field.name)
      if not _is_static_field(field):
        self._assert_no_tracer(v1, v2)
        assert "kd_field_merger" in field.metadata
        merger = field.metadata["kd_field_merger"]
        merged_fields[field.name] = merger.merge(v1, v2, updated_self)

    return dataclasses.replace(updated_self, **merged_fields)

  def _assert_no_tracer(self, v1: Any, v2: Any):
    if isinstance(v1, jax.core.Tracer) or isinstance(v2, jax.core.Tracer):
      raise RuntimeError(
          f"Tracer detected! {self.__class__.__name__}.merge should not be JIT"
          " compiled."
      )

  # Return `_SelfT` so auto-complete works
  def compute(self: _SelfT) -> _SelfT:
    """Computes final metrics from intermediate values."""
    # NOTE: this calls finalize on datafields which converts them to np.ndarrays
    return _AutoStateOutput(**{  # pytype: disable=bad-return-type
        f.name: f.metadata["kd_field_merger"].finalize(getattr(self, f.name))
        for f in dataclasses.fields(self)
        if not _is_static_field(f)
    })


# TODO(klausg): overloaded type annotation similar to dataclasses.field?


def static_field(default: Any = dataclasses.MISSING, **kwargs):
  """Define an AutoState static field.

  Static fields are not merged, and instead are checked for equality during
  `merge`. They are also not pytree nodes, so they are not touched by jax
  transforms (but can lead to recompilation if changed).
  These can be useful to store some parameters of the metric, e.g. the number
  of elements to keep. Note that static fields are rarely needed, since it is
  usually better to define static params in the corresponding metric and access
  them through the `parent` field.

  Args:
    default: The default value of the field.
    **kwargs: Additional arguments to pass to the dataclasses.field.

  Returns:
    A dataclasses.Field instance with additional metadata that marks this field
    as a static field.
  """
  metadata = kwargs.pop("metadata", {})
  metadata = metadata | {
      "pytree_node": False,
  }
  return dataclasses.field(default=default, metadata=metadata, **kwargs)


def sum_field(
    *,
    default: Any = dataclasses.MISSING,
    **kwargs,
):
  """Define an AutoState data-field that is merged by summation (a + b).

  Preserves shape and assumes that the other (merged) field has the same shape.

  Usage:

  ```python
  @flax.struct.dataclass
  class ShapePreservingAverage(AutoState):
    summed_values: Float['*any'] = sum_field()
    total_values: Float['*any'] = sum_field()

    def compute(self):
      return self.summed_values / self.total_values
  ```

  Args:
    default: The default value of the field.
    **kwargs: Additional arguments to pass to the dataclasses.field.

  Returns:
    A dataclasses.Field instance with additional metadata that marks this field
    as a pytree_node for jax and sets the field merger to _ReduceSum().
  """
  metadata = kwargs.pop("metadata", {})
  metadata = metadata | {
      "pytree_node": True,
      "kd_field_merger": _ReduceSum(),
  }
  return dataclasses.field(default=default, metadata=metadata, **kwargs)


def concat_field(
    *,
    axis: int = 0,
    default: Any = dataclasses.MISSING,
    **kwargs,
):
  """Defines a AutoState data-field that is merged by concatenation.

  During merge the data is converted to numpy and kept in a tuple of arrays.
  That way this data does not take up memory on device.
  The final compute() method concatenates the arrays along the given axis.

  Usage:

  ```python
  @flax.struct.dataclass
  class CollectTokens(AutoState):
    # merged along token axis ('n') by concatenation
    tokens: Float['b n d'] = concat_field(axis=1)
  ```

  Args:
    axis: The axis along which to concatenate the two arrays. Defaults to 0.
    default: The default value of the field.
    **kwargs: Additional arguments to pass to the dataclasses.field.

  Returns:
    A dataclasses.Field instance with additional metadata that marks this field
    as a pytree_node for jax and sets the field merger to
    _Concatenate(axis=axis).
  """
  metadata = kwargs.pop("metadata", {})
  metadata = metadata | {
      "pytree_node": True,
      "kd_field_merger": _Concatenate(axis=axis),
  }
  return dataclasses.field(default=default, metadata=metadata, **kwargs)


def truncate_field(
    *,
    num_field: str,
    axis: int | None = 0,
    default: Any = dataclasses.MISSING,
    **kwargs,
) -> Any:
  """Defines a AutoState data-field that is merged by truncation.

  During merge the data is converted to numpy and concatenated along the given
  axis. It is then truncated to the number of elements given by the `num_field`
  of its state. Useful for metrics that need to collect the first few elements
  of a tensor, e.g. the first few images for plotting.

  Usage:

  ```python
  @flax.struct.dataclass
  class CollectFirstKImages(AutoState):
    num_images: int
    images: Float['n h w 3'] = truncate_field(num_field="num_images")
  ```

  Args:
    num_field: The name of the field (in the state) that determines the number
      of elements to keep.
    axis: The axis along which to concatenate and truncate the two arrays.
      Defaults to 0.
    default: The default value of the field.
    **kwargs: Additional arguments to pass to the dataclasses.field.

  Returns:
    A dataclasses.Field instance with additional metadata that marks this field
    as a pytree_node for jax and sets the field merger to
    _Truncate(axis=axis, num_field=num_field).
  """
  # TODO(klausg): could also support setting a `num: int` directly here.
  metadata = kwargs.pop("metadata", {})
  metadata = metadata | {
      "pytree_node": True,
      "kd_field_merger": _Truncate(axis=axis, num_field=num_field),
  }
  return dataclasses.field(default=default, metadata=metadata, **kwargs)


class _FieldMerger(abc.ABC):
  """Abstract base class defining the interface for merging data-fields."""

  @abc.abstractmethod
  def merge(
      self,
      v1: Array | Empty | None,
      v2: Array | Empty | None,
      state: base_state.State,
  ) -> Array | Empty | None:
    ...

  def finalize(self, v: Array | Empty | None) -> np.ndarray | None:
    # by default convert to numpy array
    if v is EMPTY or v is None:
      return None
    return np.asarray(v)


class _ReduceSum(_FieldMerger):
  """Merges two data-fields by summing them (assumes identical shape).

  NOTE: This merger does not convert to numpy arrays. The rationale is that the
  memory consumption for fields of this type is constant, so the data can remain
  on device until the final compute() is called.
  """

  def merge(
      self,
      v1: Array | Empty | None,
      v2: Array | Empty | None,
      state: base_state.State,
  ) -> Array | Empty:
    if v1 is EMPTY:
      return v2
    if v2 is EMPTY:
      return v1
    if v1 is None or v2 is None:
      if not (v1 is None and v2 is None):
        raise ValueError("Cannot sum None and non-None values.")
      return None
    return v1 + v2


@dataclasses.dataclass(kw_only=True, frozen=True)
class _Concatenate(_FieldMerger):
  """Merges two data-fields by concatenating them along the first dimension."""

  axis: int | None = 0

  def merge(
      self,
      v1: Array | Empty | None | tuple[Array, ...],
      v2: Array | Empty | None | tuple[Array, ...],
      state: base_state.State,
  ) -> tuple[Array, ...] | None:
    if v1 is None or v2 is None:
      if not (v1 is None and v2 is None):
        raise ValueError("Cannot concatenate None and non-None values.")
      return None

    v1 = _normalize_to_tuple(v1)
    v2 = _normalize_to_tuple(v2)
    return v1 + v2  # concatenated tuples

  def finalize(
      self, v: Array | Empty | None | tuple[Array, ...]
  ) -> Array | None:
    if v is EMPTY or v is None:
      return None
    v = _normalize_to_tuple(v)
    return np.concatenate(v, axis=self.axis)


def _normalize_to_tuple(
    v: Array | Empty | tuple[Array, ...],
) -> tuple[np.ndarray, ...]:
  if v is EMPTY:
    return ()
  if not isinstance(v, tuple):
    v = (np.asarray(v),)
  return v


@dataclasses.dataclass(kw_only=True, frozen=True)
class _Truncate(_FieldMerger):
  """Merges two values by concatenating and then truncating them.

  Attributes:
    axis: The axis along which to concatenate the two arrays. Defaults to 0.
    num_field: The name of the field that contains the number of elements to
      keep. Can be any valid kontext.Path (e.g. "parent.num_images").
  """

  axis: int | None = 0
  num_field: str

  def merge(
      self,
      v1: Array | Empty | None,
      v2: Array | Empty | None,
      state: base_state.State,
  ) -> Array | Empty | None:
    num = kontext.get_by_path(state, self.num_field)
    v1 = self._maybe_truncate(v1, num)
    v2 = self._maybe_truncate(v2, num)
    if v1 is EMPTY:
      return v2
    if v2 is EMPTY:
      return v1

    if v1 is None or v2 is None:
      if not (v1 is None and v2 is None):
        raise ValueError(
            "Cannot concatenate (& truncate) None and non-None values."
        )
      return None

    assert isinstance(v1, Array) and isinstance(v2, Array)
    if v1.shape[self.axis] < num:
      v1 = np.concatenate([v1, v2], axis=self.axis)
    return self._maybe_truncate(v1, num)

  def _maybe_truncate(self, v: Array | Empty, num: int) -> Array | Empty:
    """If v is not None, then truncate it to num elements along axis."""
    if v is EMPTY or v is None:
      return v
    assert isinstance(v, Array)
    axis = np.lib.array_utils.normalize_axis_index(self.axis, v.ndim)
    return np.asarray(v[(slice(None),) * axis + (slice(None, num),)])


def _assert_static_field_equal(
    v1: Any, v2: Any, field: dataclasses.Field[Any]
) -> Any:
  """Ensure non-data fields (e.g. parent) are equal when merging."""
  # first check for empty fields to always allow merging with the empty() state.
  if v1 is EMPTY:
    return v2
  if v2 is EMPTY:
    return v1

  if v1 != v2:
    raise ValueError(
        f"Trying to merge state with different {field.name}: {v1} != {v2}"
    )
  return v1


def _is_static_field(field: dataclasses.Field[Any]) -> bool:
  return not field.metadata.get("pytree_node", False)


class _AutoStateOutput(types.SimpleNamespace):
  pass
