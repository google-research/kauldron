# Copyright 2026 The kauldron Authors.
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

"""Array types for ktyping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Sequence, TypeGuard

from etils.enp import lazy  # pylint: disable=g-importing-member
from kauldron.ktyping import dtypes
from kauldron.ktyping import internal_typing
from kauldron.ktyping import scope
from kauldron.ktyping import shape_spec_parser

MISSING = internal_typing.MISSING
Missing = internal_typing.Missing
DimValues = internal_typing.DimValues
CandidateDims = internal_typing.CandidateDims

_ArrayType = type[Any]


# MARK: ArrayTypeMeta
class ArrayTypeMeta(type):
  """Metaclass for creating array types with shape and dtype constraints.

  Usage:
    UInt8 = ArrayTypeMeta("UInt8", dtype=dtypes.uint8)

  Attributes:
    array_types: Tuple of array types that this annotation matches. E.g.
      (np.ndarray, jax.Array).
    dtype: The DType that this annotation matches. Should be a ktyping.DType.
    shape_spec: the shape specification of the arrays that this type can be
      assigned to.
  """

  _array_types: tuple[_ArrayType, ...] | Missing
  dtype: dtypes.DType | Missing
  shape_spec: str | Missing

  def __new__(
      mcs,
      name: str,
      *,
      array_types: tuple[_ArrayType, ...] | Missing = MISSING,
      dtype: dtypes.DType | Missing = MISSING,
      shape_spec: str | Missing = MISSING,
  ):
    cls_attrs = {
        "_array_types": array_types,
        "dtype": dtype,
        "shape_spec": shape_spec,
    }
    return super().__new__(mcs, name, (), cls_attrs)

  def __init__(cls, *args, **kwargs):
    del args, kwargs  # unused
    super().__init__(cls)

  @property
  def array_types(cls) -> tuple[_ArrayType, ...]:
    # This layer of indirection is here to support the long-form array type
    # specifier for the default array types: e.g. Float32[np.ndarray, "a b"]
    if cls._array_types is MISSING:
      return (NpArray, JaxArray)
    return cls._array_types  # pytype: disable=bad-return-type

  def __getitem__(
      cls,
      args: (
          str  # Array["a b"]
          | tuple[_ArrayType, str]  # Array[np.ndarray, "a b"]
          | tuple[tuple[_ArrayType, ...], str]  # Array[(np, jax), "a b"]
          | tuple[_ArrayType, str, dtypes.DType]  # Array[np, "a b", np.float32]
          | tuple[tuple[_ArrayType, ...], str, dtypes.DType]
      ),
  ) -> ArrayTypeMeta:
    """Item access syntax is used to create new array types.

    Supports three different forms:
      * Short-form: `Array["a b"]`
        When passing only the shape spec, the default or pre-set array types
        and dtypes are used.
      * Long-form: `Array[NdArray, "*b"]` / `Array[(NdArray, JaxArray), "*b"]`
        Passing the allowed array type(s) as the first argument and the
        shape spec as the second argument. Default or pre-set dtypes are used.
      * Customize dtype: Array[np.ndarray, "a b", np.float32 | np.uint8]
        Passing the allowed array type(s) as the first argument, the shape
        spec as the second argument and the dtype as the third argument.

    Args:
      args: The arguments used to construct new array type. Can be either
        `shape_spec`, `(array_types, shape_spec)`, or `(array_types, shape_spec,
        dtype)`. See above for details.

    Returns:
      A new array type with the updated shape spec, array types and dtype.

    Raises:
      TypeError: If the args do not conform to the above syntax, or if
      attempting to redefine the shape, array types or dtype of an existing
      array type.
    """

    match args:
      case str(shape_spec):
        # Short form: Array["a b"]
        name = f"{cls.__name__}[{shape_spec!r}]"
        array_types = MISSING
        dtype = MISSING
      case (array_types, str(shape_spec)):
        # Long form: Array[np.ndarray, "a b"]
        name = f"{cls.__name__}[{array_types!r}, {shape_spec!r}]"
        if not isinstance(array_types, tuple):
          array_types = (array_types,)
        dtype = MISSING
      case (array_types, str(shape_spec), dtype):
        # Long form: Array[np.ndarray, "a b", np.float32]
        name = f"{cls.__name__}[{array_types!r}, {shape_spec!r}, {dtype!r}]"
        if isinstance(array_types, tuple):
          array_types = (array_types,)
      case _:
        name = cls.__name__
        raise TypeError(
            f"Invalid get-item args: {args}. Expected one of the following:"
            f" {name}[shape_spec], {name}[array_types, shape_spec], or"
            f" {name}[array_types, shape_spec, dtype]"
        )

    if cls.shape_spec is not MISSING:
      raise TypeError(
          f"Trying to redefine shape of {cls.__name__} with {shape_spec=}."
      )
    if cls._array_types is not MISSING and array_types is not MISSING:
      raise TypeError(
          f"Trying to redefine array types of {cls.__name__} with"
          f" {array_types=}."
      )
    if cls.dtype is not MISSING and dtype is not MISSING:
      raise TypeError(
          f"Trying to redefine dtype of {cls.__name__} with {dtype=}."
      )

    return ArrayTypeMeta(
        name=name,
        array_types=(
            array_types if array_types is not MISSING else cls._array_types
        ),
        dtype=dtype if dtype is not MISSING else cls.dtype,
        shape_spec=shape_spec,
    )

  def __instancecheck__(cls, instance: Any) -> bool:
    """Check if the instance matches the constraints of this array type.

    This adds support for checks like `isinstance(arr, Float32["1 2 3"])`.

    First checks array_types, then dtypes and finally the shape using the
    currently active `ShapeScope` if it exists (fallback to an empty scope).

    Note: Unlike in jaxtyping this check does not modify the scope to keep
    isinstance checks side-effect free.

    Args:
      instance: The instance to check.

    Returns:
      True if the instance matches the constraints of this array type.
    """
    if not cls.array_types_match(instance):
      return False
    if not cls.dtype_matches(instance):
      return False
    # shape check against current dim value candidates
    if not scope.is_scope_stack_empty():
      candidates = scope.get_current_scope(nested_ok=True).candidates
    else:
      candidates = frozenset([DimValues()])

    # succeed if any of the candidates would allow this shape
    # unlike jaxtyping we do not modify the scope here to keep isinstance checks
    # side-effect free
    if cls.shape_matches(instance, candidates):
      return True
    return False

  def array_types_match(cls, instance: Any) -> bool:
    return isinstance(instance, cls.array_types)

  def dtype_matches(cls, instance: Any) -> bool:
    if cls.dtype is MISSING:
      return True
    return cls.dtype.matches(instance)  # pytype: disable=attribute-error

  def shape_matches(
      cls,
      instance: Any,
      candidates: CandidateDims = frozenset([DimValues()]),
      fstring_locals: Mapping[str, Any] | None = None,
  ) -> CandidateDims:
    """Check if the shape of the instance matches any of the candidate dims."""
    if cls.shape_spec is MISSING:
      return candidates
    if not hasattr(instance, "shape"):
      if cls.shape_spec == "":  # pylint: disable=g-explicit-bool-comparison
        return candidates  # Special case for ScalarLike types.
      else:
        return frozenset()

    shape_spec = cls.shape_spec
    if "{" in cls.shape_spec:  # has f-string syntax in shape spec -> eval
      shape_spec = eval("f" + repr(shape_spec), {}, fstring_locals)  # pylint: disable=eval-used
    spec = shape_spec_parser.parse(shape_spec)

    return spec.match(instance.shape, candidates=candidates)

  def __repr__(cls):
    return cls.__name__

  def __or__(cls, other: Any):
    """Support for annotation like `x: (Float|Int)["b h w"]`."""
    if (
        isinstance(other, ArrayTypeMeta)
        and cls.shape_spec == MISSING
        and other.shape_spec == MISSING
    ):
      if cls._array_types is MISSING:
        array_types = other.array_types
      elif other._array_types is MISSING:
        array_types = cls.array_types
      else:
        array_types = cls.array_types + other.array_types

      if cls.dtype is MISSING:
        dtype = other.dtype
      else:
        dtype = cls.dtype | other.dtype

      return ArrayTypeMeta(
          name=f"({cls.__name__}|{other.__name__})",
          array_types=array_types,
          dtype=dtype,
      )
    else:
      return type.__or__(cls, other)  # pytype: disable=unsupported-operands

  def __call__(cls, *args, **kwargs):
    """Raises a RuntimeError to prevent accidental Array("b n") syntax."""
    args_str = ", ".join(repr(a) for a in args)
    raise RuntimeError(
        f"{cls.__name__} cannot be instantiated. Did you mean to write"
        f" {cls.__name__}[{args_str}]?"
    )


# MARK: ShapeMeta
class ShapeMeta(type):
  """Metaclass for creating shape types."""

  def __new__(mcs, name: str):
    cls_attrs = {}
    return super().__new__(mcs, name, (), cls_attrs)

  def __init__(cls, *args, **kwargs):
    del args, kwargs  # unused
    super().__init__(cls)

  def __instancecheck__(cls, instance: Any) -> bool:
    # TODO(klausg): support symbolic dimensions from jax.export ?
    # TODO(klausg): support None?
    # TODO(klausg): check for positivity?
    if not isinstance(instance, Sequence):
      return False
    if not all(isinstance(s, int) for s in instance):
      return False
    return True

  def __call__(cls, *args, **kwargs):
    """Raises a RuntimeError to prevent accidental Shape("b n") syntax."""
    args_str = ", ".join(repr(a) for a in args)
    raise RuntimeError(
        f"{cls.__name__} cannot be instantiated / called. If you are migrating"
        f" from kauldron.typing please use shape({args_str}) instead."
    )

  def __repr__(cls):
    return cls.__name__


# MARK: _LazyArrayMeta
class _LazyArrayMeta(type):
  """Metaclass for array types with shape and dtype."""

  _lazy_is_array_check: Callable[[Any], bool]

  def __new__(mcs, name: str, *, lazy_is_array_check: Callable[[Any], bool]):
    bases = ()
    cls_attrs = {
        "_lazy_is_array_check": lazy_is_array_check,
    }
    return super().__new__(mcs, name, bases, cls_attrs)

  def __init__(cls, *args, **kwargs):
    del args, kwargs  # unused
    super().__init__(cls)

  def __instancecheck__(cls, obj: Any) -> bool:
    return cls._lazy_is_array_check(obj)


# MARK: Helpers
def is_array_type(origin_type: Any) -> TypeGuard[ArrayTypeMeta]:
  return isinstance(origin_type, ArrayTypeMeta)


def _is_scalar_like_type(obj: Any) -> bool:
  """Returns True if object is either an array or a python scalar."""
  is_py_scalar = isinstance(obj, (int, float, complex, bool))
  return is_py_scalar or lazy.is_array(obj)

# pylint: disable=invalid-name   # Why are they treated as constants by pylint?
NpArray = _LazyArrayMeta("NpArray", lazy_is_array_check=lazy.is_np)
JaxArray = _LazyArrayMeta("JaxArray", lazy_is_array_check=lazy.is_jax)
TfArray = _LazyArrayMeta("TfArray", lazy_is_array_check=lazy.is_tf)
TorchArray = _LazyArrayMeta("TorchArray", lazy_is_array_check=lazy.is_torch)
ScalarLike = _LazyArrayMeta(
    "ScalarLike", lazy_is_array_check=_is_scalar_like_type
)
# pylint: enable=invalid-name
