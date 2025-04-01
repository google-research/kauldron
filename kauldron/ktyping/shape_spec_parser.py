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

"""Parser for shape specs."""

import functools
from typing import Any
import warnings

from etils import epath
from kauldron.ktyping import shape_spec

# Silence deprecation warnings about sre_parse and sre_constants
with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
  import lark  # pylint: disable=g-import-not-at-top


class _ShapeSpecTransformer(lark.Transformer):
  """Transform a lark.Tree into a ShapeSpec."""

  def start(self, args: list[shape_spec.DimSpec]) -> shape_spec.ShapeSpec:
    return shape_spec.ShapeSpec(*args)

  def int_dim(self, args: list[Any]) -> shape_spec.IntDim:
    return shape_spec.IntDim(value=int(args[0]))

  def name_dim(self, args: list[Any]) -> shape_spec.NamedDims:
    return shape_spec.NamedDims(name=str(args[0]), length=1)

  def var_dim(self, args: list[Any]) -> shape_spec.NamedDims:
    return shape_spec.NamedDims(name=str(args[0]), length=None)

  def plus_dim(self, args: list[Any]) -> shape_spec.NamedDims:
    return shape_spec.NamedDims(name=str(args[0]), length=(1, None))

  def anon_dim(self, args: list[Any]) -> shape_spec.AnonDims:
    name = str(args[0]) if args else None
    return shape_spec.AnonDims(name=name, length=1)

  def anon_var_dim(self, args: list[Any]) -> shape_spec.AnonDims:
    name = str(args[0]) if args else None
    return shape_spec.AnonDims(name=name, length=None)

  def broadcast_int_dim(self, args: list[Any]) -> shape_spec.IntDim:
    return shape_spec.IntDim(value=int(args[0]), broadcastable=True)

  def broadcast_dim(
      self,
      args: list[Any],
  ) -> shape_spec.NamedDims | shape_spec.IntDim:
    return shape_spec.NamedDims(name=args[0], broadcastable=True, length=1)

  def broadcast_var_dim(self, args: list[Any]) -> shape_spec.NamedDims:
    return shape_spec.NamedDims(name=args[0], broadcastable=True, length=None)

  def broadcast_plus_dim(self, args: list[Any]) -> shape_spec.NamedDims:
    return shape_spec.NamedDims(
        name=args[0], broadcastable=True, length=(1, None)
    )

  def binary_op(self, args: list[Any]) -> shape_spec.BinaryOpDim:
    left, op, right = args
    return shape_spec.BinaryOpDim(
        left=left, right=right, op=shape_spec.SYMBOL_2_OPERATOR[str(op)]
    )

  def or_op(self, args: list[Any]) -> shape_spec.ChoiceDim:
    left, right = args
    return shape_spec.ChoiceDim(left=left, right=right)

  def optional_dim(self, args: list[Any]) -> shape_spec.OptionalDim:
    return shape_spec.OptionalDim(child=args[0])

  def neg(self, args: list[Any]) -> shape_spec.NegatedDim:
    return shape_spec.NegatedDim(child=args[0])

  def func(self, args: list[Any]) -> shape_spec.FunctionDim:
    name, arguments = args
    return shape_spec.FunctionDim(
        name=name, fn=shape_spec.NAME_2_FUNC[name], arguments=arguments
    )

  def arg_list(self, args: list[Any]) -> list[Any]:
    return args


_shape_transformer = _ShapeSpecTransformer()


class ShapeSpecSyntaxError(Exception):
  """Raised when a shape spec cannot be parsed."""

  label: str = "Shape spec syntax error"
  explanation: str = ""
  examples: tuple[str, ...] = ()

  def __str__(self):
    context, column, expected = self.args  # pylint: disable=unbalanced-tuple-unpacking
    msg = f"{self.label} at position {column}. {self.explanation}\n{context}"
    if expected:
      msg += "\nParser expected one of: \n\t* {}\n".format(
          "\n\t* ".join(expected)
      )
    return msg


class WhiteSpaceError(ShapeSpecSyntaxError):
  """Raised when a shape spec contains unexpected whitespace."""

  label = "Unexpected whitespace"
  explanation = (
      "Whitespace is only allowed between dimensions. E.g. 'a b' is allowed,"
      " but 'a + 1' is not, and einops-style '(a b)' is also not allowed."
  )
  examples = (
      "a + 1",
      "a - b",
      "a / b",
      "a * b",
      "a b + c",
      "a b - 4",
      "a b / 4",
      "(a b)",
      "a (b c)",
      "a b * c",
      "sum(a, b)",
      "a sum(b, c)",
  )


class PostModifierError(ShapeSpecSyntaxError):
  """Raised when a shape spec contains an invalid comma."""

  label = "Dimension modifier after variable"
  explanation = (
      "Dimension modifiers like * or # should precede the variable name. E.g."
      " use '*b' and '#c' and NOT 'b*' or 'c#'."
  )
  examples = (
      "a* b",
      "a b* c",
      "a# b",
      "a b# c",
      "a*# b",
      "*a# b",
      "a b*# c",
      "a *b# c",
      "sum(a*, b) c",
      "a sum(b*, c) d",
  )


class InvalidCommaError(ShapeSpecSyntaxError):
  """Raised when a shape spec contains an invalid comma."""

  label = "Invalid comma"
  explanation = "Dimensions should be separated by whitespace, not commas."
  examples = (
      "a,b",
      "a , b",
      "a b, c",
  )


class AnonymousFixedDimError(ShapeSpecSyntaxError):
  label = "Fixed anonymous dimension"
  suffix = (
      "Dimensions cannot be both fixed and anonymous. E.g. '_4' is not allowed."
  )
  examples = (
      "_4",
      "a _9",
  )


class AnonymousBroadcastableDimError(ShapeSpecSyntaxError):
  label = "Anonymous broadcastable dimension"
  explanation = (
      "Dimensions cannot be both anonymous and broadcastable. E.g. '_#a' is not"
      " allowed since it would be fully equivalent to '_a'."
  )
  examples = (
      "_#a",
      "#_a",
      "a _#b",
      "a #_b",
  )


@functools.cache
def _shape_parser() -> lark.Lark:
  grammar_path = (
      epath.resource_path("kauldron.ktyping") / "shape_spec_grammar.lark"
  )
  return lark.Lark(
      parser="lalr",
      grammar=grammar_path.read_text(),
  )


def _match_error_by_example(exc: lark.UnexpectedInput):
  errors = [
      WhiteSpaceError,
      InvalidCommaError,
      AnonymousFixedDimError,
      AnonymousBroadcastableDimError,
      PostModifierError,
  ]
  return exc.match_examples(
      _shape_parser().parse,
      {err: err.examples for err in errors},
      token_type_match_fallback=True,
  )


def _get_and_format_context(u: lark.UnexpectedInput, spec: str) -> str:
  context = u.get_context(spec).splitlines()
  assert len(context) == 2, f"Unexpected {context=}"
  prefix = "Shape spec: '"
  indent = " " * len(prefix)
  return f"{prefix}{context[0]}'\n{indent}{context[1]}"


def _get_expected_tokens(u: lark.UnexpectedInput) -> list[str]:
  expected = getattr(u, "expected", ())
  if not expected:
    expected = getattr(u, "accepts", ())
  if not expected:
    expected = getattr(u, "allowed", ())

  terminals_by_name = getattr(u, "_terminals_by_name", {})
  return sorted([
      terminals_by_name[t_name].user_repr()
      if t_name in terminals_by_name
      else t_name
      for t_name in expected
  ])


def parse(spec: str) -> shape_spec.ShapeSpec:
  """Parse a shape spec string into a ShapeSpec.

  Args:
    spec: The shape spec string to parse.

  Returns:
    The ShapeSpec object corresponding to the parsed shape spec.

  Raises:
    ShapeSpecSyntaxError: If the shape spec cannot be parsed. If possible a more
    specific error subclass is raised.
  """
  # The top-level rule of the grammar is deliberately kept as simple as possible
  # to make the state_stack of the lark.UnexpectedInput object simpler.
  # That stack is the main information used by _match_error_by_example to
  # provide better error messages.
  # Thus a simpler grammar let's us cover more cases with fewer examples.

  # But that also means that the grammar requires trailing whitespace, cannot
  # handle leading whitespace, and cannot handle empty strings.
  # So we handle those cases here.
  spec = spec.strip()
  if not spec:  # handle empty shape_spec
    return shape_spec.ShapeSpec()
  spec = spec + " "  # add trailing whitespace

  try:
    tree = _shape_parser().parse(spec)
  except lark.UnexpectedInput as u:
    exc_class = _match_error_by_example(u)
    if exc_class:
      raise exc_class(_get_and_format_context(u, spec), u.column, ()) from u
    else:
      raise ShapeSpecSyntaxError(
          _get_and_format_context(u, spec),
          u.column,
          _get_expected_tokens(u),
      ) from u
  return _shape_transformer.transform(tree)
