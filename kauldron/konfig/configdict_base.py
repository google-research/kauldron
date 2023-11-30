# Copyright 2023 The kauldron Authors.
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

"""Base ConfigDict class."""

from __future__ import annotations

import builtins
from collections.abc import Callable
import copy
import dataclasses
import functools
import itertools
import os
from typing import Any, ClassVar, Generic, TypeVar

from etils import epy
from kauldron.konfig import configdict_proxy
from kauldron.konfig import utils
import ml_collections

_T = TypeVar('_T')
_SelfT = TypeVar('_SelfT')

_ALIASES = {
    'numpy': 'np',
    'tensorflow': 'tf',
    'flax.linen': 'nn',
}

# This map the qualname to the default values to inject, when the `ConfigDict`
# is created. See `konfig.register_default_values`
_QUALNAME_TO_DEFAULT_VALUES: dict[str, ConfigDict] = {}


class ConfigDict(ml_collections.ConfigDict):
  """Wrapper around ConfigDict."""

  def __init__(
      self,
      init_dict: dict[str, Any] | ml_collections.ConfigDict | None = None,
      # Internally, `ConfigDict` call `type(self)(d, self.type_safe)`, so we
      # have to keep this, but this should never be changed
      type_safe: bool = True,
      convert_dict: bool = True,
  ) -> None:
    assert type_safe
    assert convert_dict
    init_dict = dict(init_dict or {})
    init_dict = _maybe_update_init_dict(init_dict)
    super().__init__(
        initial_dictionary=init_dict,
        type_safe=True,
        convert_dict=True,
    )

  def __getitem__(self, key: str | int) -> Any:
    key = self._normalize_arg_key(key)
    return super().__getitem__(key)

  def __setitem__(self, key: str | int, value: Any) -> None:
    key = self._normalize_arg_key(key, can_append=True)
    _assert_config_only_value(value, key)
    return super().__setitem__(key, value)

  def __repr__(self) -> str:
    visited = _VisitedTracker()
    inner_repr = visited.build_repr(self)
    return f'<ConfigDict[{inner_repr}]>'

  __str__ = __repr__

  def _repr_html_(self) -> str:
    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return ecolab.highlight_html(repr(self))

  def _normalize_arg_key(self, key: str | int, can_append: bool = False) -> str:
    """Normalize the argument key."""
    if isinstance(key, int):
      num_args = configdict_proxy.num_args(self)
      if key < 0:
        key = num_args + key
      if not (0 <= key < num_args + can_append):
        raise IndexError(f'argument index {key} is not smaller than {num_args}')
      key = str(key)
    return key

  def to_json(self, **dumps_kwargs) -> str:  # pytype: disable=signature-mismatch
    return super().to_json(utils.DefaultJSONEncoder, **dumps_kwargs)

  @property
  def ref(self: _SelfT) -> _SelfT:
    """Lazy reference access.

    Before:

    ```python
    cfg.get_ref('workdir')
    ```

    After:

    ```python
    cfg.ref.workdir
    ```

    Raises:
      RuntimeError: When used outside of a `konfig.ConfigDict` context.
    """
    return super().ref  # pytype: disable=attribute-error


def _maybe_update_init_dict(init_dict: dict[str, Any]) -> dict[str, Any]:
  """Initialize the ConfigDict with the provided values."""
  qualname = init_dict.get(configdict_proxy.QUALNAME_KEY)
  if not qualname or qualname not in _QUALNAME_TO_DEFAULT_VALUES:
    return init_dict

  # Merge the 2 dict togethers
  default_values = _QUALNAME_TO_DEFAULT_VALUES[qualname]
  default_values = copy.deepcopy(default_values)
  default_values.update(init_dict)
  return default_values


@dataclasses.dataclass
class _Visitor(Generic[_T]):
  """Recurse into a specific object type.

  Attributes:
    CLS: The type which match this
    tracker: Cycle tracker
    recurse: Function to recurse leaves
  """

  CLS: ClassVar[type[_T]]

  tracker: _VisitedTracker
  recurse: Callable[[Any], str]

  @classmethod
  def match(cls, obj: Any) -> bool:
    """Returns True if the object should be processed by the visitor."""
    return isinstance(obj, cls.CLS)

  def watch(self, obj: _T) -> Any:
    """Track whether the object was already visited or not."""
    if self.tracker.track_if_visited(obj):  # Do not recurse in cycles
      return None
    else:
      return self._recurse(obj)

  def repr(self, obj: _T) -> str:
    id_, was_repr = self.tracker.get_id_and_was_repr(obj)
    if id_:  # There's duplicate
      if not was_repr:  # First time it is displayed
        return f'&id{id_:03} ' + self._repr(obj)
      else:  # Other times, only print the reference
        return f'*id{id_:03}'
    else:  # No duplicate, only print the object
      return self._repr(obj)

  def _recurse(self, obj: _T) -> Any:
    raise NotImplementedError()

  def _repr(self, obj: _T) -> str:
    raise NotImplementedError()


class _DictVisitor(_Visitor):
  """Recurse into dict."""

  CLS = (dict, ml_collections.ConfigDict)

  def _recurse(self, obj: ml_collections.ConfigDict) -> Any:
    if isinstance(obj, dict):
      items = obj.items()
    else:
      items = obj.items(preserve_field_references=True)
    return {k: _Repr(self.recurse(v)) for k, v in items}

  def _repr(self, obj: ml_collections.ConfigDict) -> str:
    if configdict_proxy.CONST_KEY in obj:
      return self._repr_const(obj)
    elif configdict_proxy.QUALNAME_KEY in obj or isinstance(obj, ConfigDict):
      return self._repr_qualname(obj)
    else:
      return self._repr_dict(obj)

  def _repr_dict(self, obj: ml_collections.ConfigDict) -> str:
    fields = self._recurse(obj)
    return epy.Lines.make_block(
        content={repr(k): v for k, v in fields.items()},
        braces='{',
        equal=': ',
    )

  def _repr_const(self, obj: ml_collections.ConfigDict) -> str:
    return _normalize_qualname(obj[configdict_proxy.CONST_KEY])

  def _repr_qualname(self, obj: ml_collections.ConfigDict) -> str:  # pytype: disable=signature-mismatch
    """Repr qualname/ConfigDict."""
    fields = self._recurse(obj)

    if configdict_proxy.QUALNAME_KEY in obj:
      header = obj[configdict_proxy.QUALNAME_KEY]
      header = _normalize_qualname(header)
      del fields[configdict_proxy.QUALNAME_KEY]
    else:
      header = type(obj).__name__

    parts = [
        fields.pop(str(arg_id))
        for arg_id in range(configdict_proxy.num_args(fields))
    ]
    parts.extend(f'{k}={v}' for k, v in fields.items())
    parts = [_Repr(v) for v in parts]

    return epy.Lines.make_block(
        header=header,
        content=parts,
    )


@dataclasses.dataclass(slots=True)
class _Repr:
  """Forward `str` in `__repr__`."""

  value: Any

  def __repr__(self) -> str:
    if isinstance(self.value, str):
      return self.value
    else:
      return repr(self.value)


class _FieldReferenceVisitor(_Visitor):
  """Recurse into FieldReference."""

  CLS = ml_collections.FieldReference

  def _recurse(self, obj: ml_collections.FieldReference) -> Any:
    # TODO(epot): Support required=True, op
    return self.recurse(obj.get())

  def _repr(self, obj: ml_collections.FieldReference) -> str:
    # The `&id000` makes it explicit already which fields are references
    return self._recurse(obj)


class _ListVisitor(_Visitor):
  """Recurse into list, tuple."""

  CLS = (list, tuple)

  def _recurse(self, obj: list[Any] | tuple[Any]) -> Any:
    return [_Repr(self.recurse(v)) for v in obj]

  def _repr(self, obj: list[Any] | tuple[Any]) -> str:
    return epy.Lines.make_block(
        content=self._recurse(obj),
        braces='[' if isinstance(obj, list) else '(',
    )


class _DefaultVisitor(_Visitor):
  """Leaves."""

  CLS = object

  def watch(self, obj: object) -> Any:
    # Do not track leaves
    return None

  def repr(self, obj: Any) -> Any:
    return self._repr(obj)

  def _recurse(self, obj: object) -> Any:
    return None

  def _repr(self, obj: object) -> str:
    if obj == ...:
      return '...'
    else:
      return repr(obj)


@dataclasses.dataclass
class _VisitedTracker:
  """Cycle tracker and detector."""

  count: itertools.count = dataclasses.field(default_factory=itertools.count)
  # Mapping id(obj) -> &001
  pyid_to_id: dict[int, int | None] = dataclasses.field(default_factory=dict)
  # Whether the object is displayed once or not
  pyid_was_repr: set[int] = dataclasses.field(default_factory=set)

  VISITORS = [
      _DictVisitor,
      _FieldReferenceVisitor,
      _ListVisitor,
      _DefaultVisitor,
  ]

  def __post_init__(self):
    next(self.count)  # Start at 1

  def build_repr(self, obj: ConfigDict) -> str:
    # First traverse the object to detect the duplicates and cycles
    self.recurse(obj, is_repr=False)

    # Then traverse to build the repr
    return self.recurse(obj, is_repr=True)

  def recurse(self, obj: Any, *, is_repr: bool) -> Any:
    """Recursivelly explore the object to detect the cycles."""

    for visitor_cls in self.VISITORS:
      if visitor_cls.match(obj):
        visitor = visitor_cls(
            tracker=self,
            recurse=functools.partial(self.recurse, is_repr=is_repr),
        )
        if is_repr:
          return visitor.repr(obj)
        else:
          return visitor.watch(obj)
    else:
      raise TypeError(f'Unexpected {obj!r}')

  def track_if_visited(self, obj: Any) -> bool:
    pyid = id(obj)
    if pyid not in self.pyid_to_id:  # Never visited, track
      self.pyid_to_id[pyid] = None
      return False
    elif self.pyid_to_id[pyid] is None:  # Already visited, set a new id
      self.pyid_to_id[pyid] = next(self.count)  # pytype: disable=container-type-mismatch
      return True
    else:  # Already visited and id set, do nothing
      return True

  def get_id_and_was_repr(self, obj) -> tuple[int, bool]:
    id_ = self.pyid_to_id[id(obj)]
    if not id_:  # Object has no duplicate
      return 0, False
    # Object has duplicate
    if id_ in self.pyid_was_repr:
      return id_, True  # Object was already repr  # pytype: disable=bad-return-type
    else:
      self.pyid_was_repr.add(id_)  # pytype: disable=container-type-mismatch
      return id_, False  # Object never repr  # pytype: disable=bad-return-type


def _normalize_qualname(name: str) -> str:
  """Normalize the qualname for nicer display."""
  for key, alias in _ALIASES.items():
    if name.startswith(key):
      name = name.replace(key, alias, 1)
  return name.replace(':', '.')


def register_aliases(aliases: dict[str, str]) -> None:
  """Register module aliases for nicer display.

  Example:

  ```python
  konfig.register_aliases({
      'jax.numpy': 'jnp',
      'tensorflow.experimental.numpy': 'tnp',
  })

  with konfig.imports()
    import jax.numpy as jnp

  assert repr(jnp.int32) == 'jnp.int32'
  # Without aliases, repr(jnp.int32) == 'jax.numpy.int32'
  ```

  Args:
    aliases: The mapping import name to display alias.
  """
  # Allow overwritten keys: For colab and for tests. Aliases are used
  # only for display, so don't really matter.
  _ALIASES.update(aliases)


def register_default_values(default_values: ConfigDict) -> None:
  """Register default values when creating the ConfigDict.

  Some class want to inject default values when being created as `ConfigDict`,
  like:

  * `cfg = kd.train.Trainer()` create `cfg.workdir = placeholder()`, so
    the user don't need to specify it in it's config
  * `cfg = kxm.Job()` create `cfg.executor = Borg()`, to allow the CLI to access
    nested fields (e.g. `job.executor.scheduling.max_task_failures = 0`) without
    having to define them in the `get_config()`.

  Usage:

  ```python
  with konfig.imports():
    import kauldron as kd

  konfig.register_default_values(
      kd.train.Trainer(
          workdir=konfig.placeholder(str),
      )
  )
  ```

  Args:
    default_values: The default `ConfigDict` to create.
  """
  qualname = default_values.__qualname__
  if registered_values := _QUALNAME_TO_DEFAULT_VALUES.get(qualname):
    raise ValueError(f'{qualname} is already registered: {registered_values}.')
  _QUALNAME_TO_DEFAULT_VALUES[qualname] = default_values


def _assert_config_only_value(value, name) -> None:
  """Validate only config values are defined."""
  # TODO(epot): Could have better error message (display path)
  # TODO(epot): Test `__setattr__` is also called in `ConfigDict({})`
  match value:
    case ConfigDict():  # Leafs should have been already validated
      return
    case ml_collections.FieldReference():
      return
    case (
        int()
        | float()
        | bool()
        | str()
        | bytes()
        | None
        | builtins.Ellipsis
        | slice()
        | os.PathLike()  # Exceptionally allow pathlib object
    ):
      return  # Built-ins
    case dict() | ml_collections.ConfigDict():
      for v in value.values():
        _assert_config_only_value(v, name)
    case list() | tuple() | set() | frozenset():
      for v in value:
        _assert_config_only_value(v, name)
    case configdict_proxy.ConfigDictProxyObject():
      return
    case _:
      raise ValueError(
          f'Error setting `cfg.{name}`: To avoid mixing configurable and'
          ' resolved python object, ConfigDict only accept other configurables'
          f' (list, int, ConfigDict,...). Got: {_shortn(value, 40)}\nYou can'
          ' use `with kd.konfig.mock_modules():` to set the value.'
      )


def _shortn(x: str, max_length: int) -> str:
  """Shortn the string."""
  if not isinstance(x, str):
    x = repr(x)
  if len(x) <= max_length:
    return x

  keep_chars = max_length // 2
  return x[:keep_chars] + '...' + x[-keep_chars:]


_Field = ml_collections.FieldReference | ConfigDict
