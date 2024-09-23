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

"""A parser for shape specs."""

from __future__ import annotations

import abc
import base64
import dataclasses
import enum
import itertools
import math
import operator
import pickle
from typing import Any, Callable, Optional
import zlib

from kauldron.typing import utils
from kauldron.utils import standalone_parser

# Serialized parser from grammar in shape_spec.lark (lark==1.2.1)
DATA = b'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'
DATA = pickle.loads(zlib.decompress(base64.b64decode(DATA)))
MEMO = b'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'
MEMO = pickle.loads(zlib.decompress(base64.b64decode(MEMO)))
_parser = standalone_parser.get_parser((DATA, MEMO))


class _Priority(enum.IntEnum):
  ADD = enum.auto()
  MUL = enum.auto()
  POW = enum.auto()
  UNARY = enum.auto()
  ATOM = enum.auto()


class DimSpec(abc.ABC):

  def evaluate(self, memo: utils.Memo) -> tuple[int, ...]:
    raise NotImplementedError()

  @property
  def priority(self) -> int:
    return _Priority.ATOM


@dataclasses.dataclass(init=False)
class ShapeSpec:
  """Parsed shape specification."""

  dim_specs: tuple[DimSpec, ...]

  def __init__(self, *dim_specs: DimSpec):
    self.dim_specs = tuple(dim_specs)

  def evaluate(self, memo: utils.Memo) -> tuple[int, ...]:
    return tuple(
        itertools.chain.from_iterable(s.evaluate(memo) for s in self.dim_specs)
    )

  def __repr__(self):
    return ' '.join(repr(ds) for ds in self.dim_specs)


@dataclasses.dataclass
class IntDim(DimSpec):
  value: int
  broadcastable: bool = False

  def evaluate(self, memo: utils.Memo) -> tuple[int, ...]:
    if self.broadcastable:
      raise utils.ShapeError(f'Cannot evaluate a broadcastable dim: {self!r}')
    return (self.value,)

  def __repr__(self):
    prefix = '_' if self.broadcastable else ''
    return prefix + str(self.value)


@dataclasses.dataclass
class SingleDim(DimSpec):
  """Simple individual dimensions like "height", "_a" or "#c"."""

  name: Optional[str] = None
  broadcastable: bool = False
  anonymous: bool = False

  def evaluate(self, memo: utils.Memo) -> tuple[int, ...]:
    if self.anonymous:
      raise utils.ShapeError(f'Cannot evaluate anonymous dimension: {self!r}')
    elif self.broadcastable:
      raise utils.ShapeError(
          f'Cannot evaluate a broadcastable dimension: {self!r}'
      )
    elif self.name not in memo.single:
      raise utils.ShapeError(
          f'No value known for {self!r}. '
          f'Known values are: {sorted(memo.single.keys())}'
      )
    else:
      return (memo.single[self.name],)

  def __repr__(self):
    return (
        ('#' if self.broadcastable else '')
        + ('_' if self.anonymous else '')
        + (self.name if self.name else '')
    )


@dataclasses.dataclass
class VariadicDim(DimSpec):
  """Variable size dimension specs like "*batch" or "..."."""

  name: Optional[str] = None
  anonymous: bool = False
  broadcastable: bool = False

  def evaluate(self, memo: utils.Memo) -> tuple[int, ...]:
    if self.anonymous:
      raise utils.ShapeError(f'Cannot evaluate anonymous dimension: {self!r}')
    if self.broadcastable:
      raise utils.ShapeError(
          f'Cannot evaluate a broadcastable variadic dimension: {self!r}'
      )
    if self.name not in memo.variadic:
      raise utils.ShapeError(
          f'No value known for {self!r}. Known values are:'
          f' {sorted(memo.variadic.keys())}'
      )
    return memo.variadic[self.name]

  def __repr__(self):
    if self.anonymous:
      return '...'
    if self.broadcastable:
      return '*#' + self.name
    else:
      return '*' + self.name


BinOp = Callable[[Any, Any], Any]


@dataclasses.dataclass
class Operator:
  symbol: str
  fn: BinOp
  priority: _Priority


OPERATORS = [
    Operator('+', operator.add, _Priority.ADD),
    Operator('-', operator.sub, _Priority.ADD),
    Operator('*', operator.mul, _Priority.MUL),
    Operator('/', operator.truediv, _Priority.MUL),
    Operator('//', operator.floordiv, _Priority.MUL),
    Operator('%', operator.mod, _Priority.MUL),
    Operator('**', operator.pow, _Priority.POW),
]

SYMBOL_2_OPERATOR = {o.symbol: o for o in OPERATORS}


@dataclasses.dataclass
class FunctionDim(DimSpec):
  """Function based dimension specs like "min(a,b)" or "sum(*batch)."""

  name: str
  fn: Callable[..., int]
  arguments: list[DimSpec]

  def evaluate(self, memo: utils.Memo) -> tuple[int]:  # pylint: disable=g-one-element-tuple
    vals = itertools.chain.from_iterable(
        arg.evaluate(memo) for arg in self.arguments
    )
    return (self.fn(vals),)

  def __repr__(self):
    arg_list = ','.join(repr(a) for a in self.arguments)
    return f'{self.name}({arg_list})'


NAME_2_FUNC = {'sum': sum, 'min': min, 'max': max, 'prod': math.prod}


@dataclasses.dataclass
class BinaryOpDim(DimSpec):
  """Binary ops for dim specs such as "H*W" or "C+1"."""

  op: Operator
  left: DimSpec
  right: DimSpec

  def evaluate(self, memo: utils.Memo) -> tuple[int]:  # pylint: disable=g-one-element-tuple
    (left,) = self.left.evaluate(memo)  # unpack tuple (has to be 1-dim)
    (right,) = self.right.evaluate(memo)  # unpack tuple (has to be 1-dim)
    return (self.op.fn(left, right),)

  @property
  def priority(self) -> int:
    return self.op.priority

  def __repr__(self):
    left_repr = (
        repr(self.left)
        if self.priority < self.left.priority
        else f'({self.left!r})'
    )
    right_repr = (
        repr(self.right)
        if self.priority < self.right.priority
        else f'({self.right!r})'
    )
    return f'{left_repr}{self.op.symbol}{right_repr}'


@dataclasses.dataclass
class NegDim(DimSpec):
  """Negation of a dim spec, e.g. "-h"."""

  child: DimSpec

  def evaluate(self, memo: utils.Memo) -> tuple[int]:  # pylint: disable=g-one-element-tuple
    return (-self.child.evaluate(memo)[0],)

  @property
  def priority(self) -> int:
    return _Priority.UNARY

  def __repr__(self):
    if self.priority < self.child.priority:
      return f'-{self.child!r}'
    else:
      return f'-({self.child!r})'


class ShapeSpecTransformer(standalone_parser.Transformer):
  """Transform a lark standalone_parser.Tree into a ShapeSpec."""

  @staticmethod
  def start(args: list[DimSpec]) -> ShapeSpec:
    return ShapeSpec(*args)

  @staticmethod
  def int_dim(args: list[Any]) -> IntDim:
    return IntDim(value=int(args[0]))

  @staticmethod
  def name_dim(args: list[Any]) -> SingleDim:
    return SingleDim(name=args[0])

  @staticmethod
  def anon_dim(args: list[Any]) -> SingleDim:
    name = args[0] if args else None
    return SingleDim(name=name, anonymous=True)

  @staticmethod
  def anon_var_dim(args: list[Any]) -> VariadicDim:
    name = args[0] if args else None
    return VariadicDim(name=name, anonymous=True)

  @staticmethod
  def var_dim(args: list[Any]) -> VariadicDim:
    return VariadicDim(name=args[0])

  @staticmethod
  def broadcast_dim(args: list[Any]) -> DimSpec:
    try:
      return IntDim(value=int(args[0]), broadcastable=True)
    except ValueError:
      return SingleDim(name=args[0], broadcastable=True)

  @staticmethod
  def broadcast_var_dim(args: list[Any]) -> VariadicDim:
    return VariadicDim(name=args[0], broadcastable=True)

  @staticmethod
  def binary_op(args: list[Any]) -> BinaryOpDim:
    left, op, right = args
    return BinaryOpDim(left=left, right=right, op=SYMBOL_2_OPERATOR[str(op)])

  @staticmethod
  def neg(args: list[Any]) -> NegDim:
    return NegDim(child=args[0])

  @staticmethod
  def func(args: list[Any]) -> FunctionDim:
    name, arguments = args
    return FunctionDim(name=name, fn=NAME_2_FUNC[name], arguments=arguments)

  @staticmethod
  def arg_list(args: list[Any]) -> list[Any]:
    return args


def parse(spec: str) -> ShapeSpec:
  tree = _parser.parse(spec)
  return ShapeSpecTransformer().transform(tree)
