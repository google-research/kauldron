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

import jax
import jax.export
from kauldron.ktyping import dim_view
from kauldron.ktyping import errors
from kauldron.ktyping import internal_typing as ktype
from kauldron.ktyping import scope as scope_mod
from kauldron.ktyping import shape_spec_parser
from kauldron.ktyping import shape_tools
from kauldron.ktyping import typeguard_checkers as tgc
from kauldron.ktyping.array_types import ArraySpec, Float  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
import numpy as np
import pytest


def _make_symbolic_dim(name: str = "B"):
  dims = jax.export.symbolic_shape(name)
  return dims[0]


class TestEntryPoint:

  def test_shape_matches_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])

  def test_named_dim_binding(self):
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      assert dims["T"] == 128

  def test_fixed_dim_with_concrete(self):
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["B 128"])


class TestShapeMatching:

  def test_named_dim_match(self):
    sym_b = _make_symbolic_dim("B")
    spec = shape_spec_parser.parse("B T D")
    shape = (sym_b, 128, 64)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) == 1
    dv = next(iter(result))
    assert dv["T"] == (128,)
    assert dv["D"] == (64,)
    assert dv["B"] == (sym_b,)

  def test_consistency_same_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    spec = shape_spec_parser.parse("B B")
    shape = (sym_b, sym_b)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) == 1

  def test_consistency_different_symbolic(self):
    # sym_b and sym_t can't be proven unequal, so they might match.
    sym_b = _make_symbolic_dim("B")
    sym_t = _make_symbolic_dim("T")
    spec = shape_spec_parser.parse("D D")
    shape = (sym_b, sym_t)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) >= 1

  def test_multi_dim_binding(self):
    sym_b = _make_symbolic_dim("B")
    spec = shape_spec_parser.parse("*batch D")
    shape = (sym_b, 128, 64)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) >= 1

  def test_binary_op_eval(self):
    sym_b = _make_symbolic_dim("B")
    dim_values = ktype.DimValues({"B": (sym_b,), "T": (128,)})
    spec = shape_spec_parser.parse("B T")
    result = spec.evaluate(dim_values)
    assert result == (sym_b, 128)

  def test_arithmetic_match(self):
    sym_b = _make_symbolic_dim("B")
    dim_values = ktype.DimValues({"h": (128,)})
    spec = shape_spec_parser.parse("B h+1")
    shape = (sym_b, 129)
    result = spec.match(shape, frozenset([dim_values]))
    assert len(result) == 1
    dv = next(iter(result))
    assert dv["B"] == (sym_b,)
    assert dv["h"] == (128,)

  def test_arithmetic_eval_symbolic(self):
    sym_h = _make_symbolic_dim("H")
    dim_values = ktype.DimValues({"H": (sym_h,)})
    spec = shape_spec_parser.parse("H+1")
    result = spec.evaluate(dim_values)
    assert len(result) == 1

  def test_arithmetic_match_symbolic_infer(self):
    sym_b = _make_symbolic_dim("B")
    sym_h = _make_symbolic_dim("H")
    spec = shape_spec_parser.parse("B h+1")
    shape = (sym_b, sym_h + 1)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) >= 1


class TestDimView:

  def test_dim_read_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      b_val = dims["B"]
      assert b_val is sym_b or b_val == sym_b

  def test_dim_write_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    with typechecked():
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      dims["B"] = sym_b
      assert dims["B"] is sym_b or dims["B"] == sym_b

  def test_dim_str_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      dims_str = str(dims)
      assert "B" in dims_str
      assert "T" in dims_str
      assert "&" in dims_str

  @pytest.mark.parametrize(
      "dim_value_fn, expected",
      [
          (lambda sym_b: (sym_b,), "&b"),
          (lambda sym_b: (128,), "128"),
          (lambda sym_b: (sym_b, 128), "(&b, 128)"),
          (lambda _: (ktype.UNKNOWN_DIM,), "#"),
          (lambda sym_b: (sym_b + 1,), "&b + 1"),
          (lambda _: (ktype.UNKNOWN_DIM, 128), "(#, 128)"),
      ],
  )
  def test_format_dim_value(self, dim_value_fn, expected):
    sym_b = _make_symbolic_dim("b")
    dim_value_tuple = dim_value_fn(sym_b)
    result = dim_view._format_dim_value(dim_value_tuple)
    assert result == expected


class TestShapeTools:

  def test_eval_shape_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    dim_values = ktype.DimValues({"B": (sym_b,), "T": (128,)})
    result = shape_tools._eval_shape("B T", frozenset([dim_values]))
    assert result[1] == 128

  def test_shape_in_typechecked(self):
    sym_b = _make_symbolic_dim("B")

    @typechecked
    def f(x: ArraySpec["B T"]):
      del x
      return shape_tools.shape("T B")

    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    result = f(arr)
    assert result[0] == 128


class TestDtypeStillChecked:

  def test_wrong_dtype_still_type_checked(self):
    arr_int = np.zeros((2, 3), dtype=np.int32)
    with typechecked():
      with pytest.raises(errors.KTypeCheckError, match="dtype"):
        tgc.check_type(arr_int, Float["B T"])


class TestIntegration:

  def test_jax_export_with_typechecked(self):
    @typechecked
    def my_fn(x: Float["B T D"]) -> Float["B T D"]:
      s = shape_tools.shape("B T D")
      b = dim_view.dim["B"]
      del s, b
      return x * 2

    (sym_b,) = jax.export.symbolic_shape("B")
    x = jax.ShapeDtypeStruct((sym_b, 128, 64), jax.numpy.float32)
    exported = jax.export.export(jax.jit(my_fn))(x)
    assert exported is not None

  def test_shape_return_type_with_symbolic_dims(self):
    from kauldron.ktyping.array_types import Shape  # pylint: disable=g-import-not-at-top,g-importing-member

    @typechecked
    def flatten(
        tokens: ArraySpec["... d"],
    ) -> tuple[ArraySpec["... d"], Shape, Shape]:
      original_shape = tuple(tokens.shape)
      return tokens, original_shape, original_shape

    (sym_b,) = jax.export.symbolic_shape("B")
    x = jax.ShapeDtypeStruct((sym_b, 128, 64), jax.numpy.float32)
    result = flatten(x)
    assert result[1] == (sym_b, 128, 64)
    assert result[2] == (sym_b, 128, 64)


class TestEdgeCases:

  def test_broadcastable_multi_dim_symbolic(self):
    sym_b = _make_symbolic_dim("B")
    spec = shape_spec_parser.parse("*b D")
    shape = (sym_b, 1, 64)
    result = spec.match(shape, frozenset([ktype.DimValues()]))
    assert len(result) >= 1

  def test_negated_dim_symbolic(self):
    sym_h = _make_symbolic_dim("H")
    dim_values = ktype.DimValues({"H": (sym_h,)})
    spec = shape_spec_parser.parse("-H")
    result = spec.evaluate(dim_values)
    assert len(result) == 1

  def test_function_dim_min_symbolic_evaluate(self):
    sym_a = _make_symbolic_dim("A")
    dim_values = ktype.DimValues({"A": (sym_a,), "B": (128,)})
    spec = shape_spec_parser.parse("min(A,B)")
    result = spec.evaluate(dim_values)
    assert len(result) == 1

  def test_function_dim_min_symbolic_match(self):
    sym_a = _make_symbolic_dim("A")
    dim_values = ktype.DimValues({"A": (sym_a,), "B": (128,)})
    spec = shape_spec_parser.parse("min(A,B)")
    shape = (64,)
    result = spec.match(shape, frozenset([dim_values]))
    assert len(result) >= 1

  def test_error_message_symbolic(self):
    # sym_b might be 5, so matching (sym_b, 128) against "5 128" should not
    # raise — we can't prove inequality.
    sym_b = _make_symbolic_dim("B")
    arr = jax.ShapeDtypeStruct((sym_b, 128), np.float32)
    with typechecked():
      tgc.check_type(arr, ArraySpec["5 128"])  # should not raise


class TestNoneDims:

  def _make_none_shape_struct(self, shape, dtype=np.float32):
    return jax.ShapeDtypeStruct(shape, dtype)

  def test_known_dims_still_checked(self):
    arr = self._make_none_shape_struct((None, 128))
    with typechecked():
      tgc.check_type(arr, ArraySpec["B 128"])

  def test_known_dims_mismatch_raises(self):
    arr = self._make_none_shape_struct((None, 128))
    with typechecked():
      with pytest.raises(errors.KTypeCheckError):
        tgc.check_type(arr, ArraySpec["B 64"])

  def test_named_dim_binding_with_none(self):
    arr = self._make_none_shape_struct((None, 128))
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      assert dims["T"] == 128

  def test_none_dim_binds_as_unknown(self):
    arr = self._make_none_shape_struct((None, 128))
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      assert dims["B"] == ktype.UNKNOWN_DIM

  def test_dim_read_works_for_known_dims(self):
    arr = self._make_none_shape_struct((None, 64, 32))
    with typechecked():
      tgc.check_type(arr, ArraySpec["B H W"])
      s = scope_mod.get_current_scope(nested_ok=True)
      dims = dim_view.DimView(s)
      assert dims["H"] == 64
      assert dims["W"] == 32

  def test_shape_eval_with_none(self):
    arr = self._make_none_shape_struct((None, 128))
    with typechecked():
      tgc.check_type(arr, ArraySpec["B T"])
      result = shape_tools.shape("T")
      assert result == (128,)

  def test_format_dim_value_none_mixed(self):
    result = dim_view._format_dim_value((ktype.UNKNOWN_DIM, 128))
    assert result == "(#, 128)"

  def test_consistent_with_none_in_shape(self):
    arr1 = self._make_none_shape_struct((None, 128))
    arr2 = self._make_none_shape_struct((None, 128))
    with typechecked():
      tgc.check_type(arr1, ArraySpec["B T"])
      tgc.check_type(arr2, ArraySpec["B T"])

  def test_broadcastable_with_none(self):
    arr = self._make_none_shape_struct((None, None, 128))
    with typechecked():
      tgc.check_type(arr, ArraySpec["*b T"])
