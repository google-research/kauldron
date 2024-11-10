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

"""Base ConfigDict class."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import functools
import itertools
import json
import os
from typing import Any, ClassVar, Generic, Self, TypeVar

from etils import epy
from kauldron.konfig import configdict_proxy
from kauldron.konfig import utils
import ml_collections

_T = TypeVar('_T')
_SelfT = TypeVar('_SelfT')

_ALIASES = {
    'numpy': 'np',
    'tensorflow_datasets': 'tfds',
    'tensorflow': 'tf',
    'flax.linen': 'nn',
}

# This map the qualname to the default values to inject, when the `ConfigDict`
# is created. See `konfig.register_default_values`
_QUALNAME_TO_DEFAULT_VALUES: dict[str, ConfigDict] = {}

# Register to support `isinstance(cfg, collections.abc.Mapping)`
MutableMapping.register(ml_collections.ConfigDict)  # pytype: disable=attribute-error


class ConfigDict(ml_collections.ConfigDict):
  """Wrapper around ConfigDict."""

  def __init__(
      self,
      init_dict: dict[str, Any] | ml_collections.ConfigDict | None = None,
      *,
      # If `True`, skip the normalization to avoid infinite recursion
      _normalized: bool = False,
  ) -> None:
    init_dict = dict(init_dict or {})
    init_dict = _maybe_update_init_dict(init_dict)  # pytype: disable=name-error

    # Normalize here rather than at the individul field level (`__setattr__`),
    # to have a global cache for all shared values (so shared fields are
    # correctly handled).
    if not _normalized:
      init_dict = _normalize_config_only_value(init_dict, '', id_to_dict={})  # pytype: disable=name-error
    super().__init__(
        initial_dictionary=init_dict,
        type_safe=True,
        # `ConfigDict` already normalize everything to `dict`, so disable
        # it here as it creates issues with `FieldReference` (accessed before
        # the value is resolved).
        convert_dict=False,
        sort_keys=False,  # Keep original key order
        allow_dotted_keys=True,
    )

  def __getitem__(self, key: str | int) -> Any:
    key = self._normalize_arg_key(key)
    return super().__getitem__(key)

  def __setitem__(self, key: str | int, value: Any) -> None:
    key = self._normalize_arg_key(key, can_append=True)
    value = _normalize_config_only_value(value, key, id_to_dict={})
    return super().__setitem__(key, value)

  def __deepcopy__(self, memo: dict[int, Any]) -> Self:
    # First create an empty copy and add it to the memo to avoid infinite
    # recursion.
    new = type(self)()
    memo[id(self)] = new

    # Then copy the content and recurse into the sub-fields.
    new.update({
        k: copy.deepcopy(v, memo)
        for k, v in self.items(preserve_field_references=True)
    })
    return new

  def __repr__(self) -> str:
    visited = _VisitedTracker()
    inner_repr = visited.build_repr(self)
    return f'<ConfigDict[{inner_repr}]>'

  __str__ = __repr__

  def _repr_html_(self) -> str:
    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return ecolab.highlight_html(repr(self))

  def _normalize_arg_key(
      self,
      key: str | int,
      can_append: bool = False,
  ) -> str | int:
    """Normalize the argument key."""
    key = utils.maybe_decode_json_key(key)

    if isinstance(key, int) and configdict_proxy.QUALNAME_KEY in self:
      num_args = configdict_proxy.num_args(self)
      if key < 0:
        key = num_args + key
      if not (0 <= key < num_args + can_append):
        raise IndexError(f'argument index {key} is not smaller than {num_args}')
      key = str(key)
    elif isinstance(key, tuple):
      # In the CLI: `--cfg.args['a.b']` is parsed as `('args', ('a', 'b')))`
      key = '.'.join(key)
    return key

  def to_json(self, **dumps_kwargs) -> str:  # pytype: disable=signature-mismatch
    return json.dumps(utils.to_json(self), **dumps_kwargs)

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


def _maybe_update_init_dict(init_dict: Mapping[str, Any]) -> Mapping[str, Any]:
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
    return {
        k: _Repr(self.recurse(v)) for k, v in _items_preserve_reference(obj)
    }

  def _repr(self, obj: ml_collections.ConfigDict) -> str:
    if configdict_proxy.CONST_KEY in obj:
      return self._repr_const(obj)
    elif configdict_proxy.QUALNAME_KEY in obj:
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
    # TODO(epot): The `id` tracking system do not work with references on
    # integers and primitives. Only sub-configs. Not sure how to support this.
    return self._recurse(obj)


class _ListVisitor(_Visitor):
  """Recurse into list, tuple."""

  CLS = (list, tuple)

  def _recurse(self, obj: list[Any] | tuple[Any, ...]) -> Any:
    return [_Repr(self.recurse(v)) for v in obj]

  def _repr(self, obj: list[Any] | tuple[Any, ...]) -> str:
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
  pyid_to_id: dict[int, utils.CachedObj[int | None]] = dataclasses.field(
      default_factory=dict
  )
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
      self.pyid_to_id[pyid] = utils.CachedObj(ref=obj, value=None)
      return False
    elif self.pyid_to_id[pyid].value is None:  # Already visited, set a new id
      self.pyid_to_id[pyid].value = next(self.count)  # pytype: disable=container-type-mismatch
      return True
    else:  # Already visited and id set, do nothing
      return True

  def get_id_and_was_repr(self, obj) -> tuple[int, bool]:
    # With `konfig.ref_fn`, some objects are re-created everytime the attribute
    # is accessed. So some objects might have new `id()`
    if not (cached := self.pyid_to_id.get(id(obj))):  # Object has no duplicate
      return 0, False
    # Object has duplicate
    id_ = cached.value
    if id_ in self.pyid_was_repr:
      return id_, True  # Object was already repr  # pytype: disable=bad-return-type
    else:
      self.pyid_was_repr.add(id_)  # pytype: disable=container-type-mismatch
      return id_, False  # Object never repr  # pytype: disable=bad-return-type


def _normalize_qualname(name: str) -> str:
  """Normalize the qualname for nicer display."""
  for key, alias in _ALIASES.items():
    if name == key or name.startswith((f'{key}.', f'{key}:')):
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


def register_default_values(default_values: utils.ConfigDictLike[Any]) -> None:
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
    from kauldron import kd

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
  if (registered_values := _QUALNAME_TO_DEFAULT_VALUES.get(qualname)) and (
      default_values != registered_values
  ):
    raise ValueError(f'{qualname} is already registered: {registered_values}.')
  _QUALNAME_TO_DEFAULT_VALUES[qualname] = default_values


def _normalize_config_only_value(value, name, *, id_to_dict) -> Any:
  """Validate only config values are defined.

  Args:
    value: The value to normalize (e.g. `dict` -> `ConfigDict`).
    name: The name of the field (for better error message)
    id_to_dict: Cache to support shared objects (`ConfigDict` created once and
      reused in multiple places).

  Returns:
    The normalized value.
  """
  # TODO(epot): Test `__setattr__` is also called in `ConfigDict({})`
  normalize_fn = functools.partial(
      _normalize_config_only_value, id_to_dict=id_to_dict
  )
  match value:
    case dict() | ml_collections.ConfigDict():
      if (id_ := value.get('__id__')) is not None:  # Shared value:
        del value['__id__']  # ConfigDict do not have `.pop()`
        if id_ in id_to_dict:  # ConfigDict already constructed
          return id_to_dict[id_]  # Reuse same instance

      if isinstance(value, ConfigDict):
        # Leafs should have been already validated but might have a `__id__`
        # if they are coming from `konfig.register_default_values()` due to
        # `default_values.update(init_dict)` above.
        cfg = value
      else:
        cfg = ConfigDict(
            {  # Convert `dict` -> `ConfigDict`
                k: normalize_fn(v, f'{name}.{k}')
                for k, v in _items_preserve_reference(value)
            },
            # Skip normalization to avoid infinite recursion. Is there a cleaner
            # way ?
            _normalized=True,
        )

      if id_ is not None:  # Save shared value
        id_to_dict[id_] = cfg
      return cfg
    case ml_collections.FieldReference():
      return value
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
      return value  # Built-ins
    case list() | tuple() | set() | frozenset():
      return type(value)(
          normalize_fn(v, f'{name}[{i}]') for i, v in enumerate(value)
      )
    case configdict_proxy.ConfigDictProxyObject():
      # TODO(epot): Rather than inheriting from dict, `ConfigDictProxyObject`
      # could simply implement the `ConfigDictConvertible` protocol
      return value
    case utils.ConfigDictConvertible():
      return normalize_fn(value.__as_konfig__(), name)
    case functools.partial():
      # TODO(epot): Cleaner way to create the object with
      # `fake_import_utils.mock_modules()` rather than manually create it ?
      return ConfigDict(
          {
              '__qualname__': 'functools:partial',
              '0': normalize_fn(value.func, f'{name}.0'),
              **{
                  str(i + 1): normalize_fn(a, f'{name}.{i+1}')
                  for i, a in enumerate(value.args)
              },
              **{
                  k: normalize_fn(v, f'{name}.{k}')
                  for k, v in value.keywords.items()
              },
          },
          # Skip normalization to avoid infinite recursion. Is there a cleaner
          # way ?
          _normalized=True,
      )
    case _:
      raise ValueError(
          f'Error setting `cfg.{name}`: To avoid mixing configurable and'
          ' resolved python object, ConfigDict only accept other configurables'
          f' (list, int, ConfigDict,...). Got: {_shortn(value, 40)}\n'
          ' * In your config, you might have forgotten to wrap the import '
          '   inside `konfig.imports()`\n'
          ' * On Colab, you can wrap the assignment in '
          '   `with kd.konfig.mock_modules()` to locally mock the module.\n'
      )


def _shortn(x: Any, max_length: int) -> str:
  """Shortn the string."""
  if not isinstance(x, str):
    x = repr(x)
  if len(x) <= max_length:
    return x

  keep_chars = max_length // 2
  return x[:keep_chars] + '...' + x[-keep_chars:]


def _items_preserve_reference(
    cfg: dict[str, Any] | ml_collections.ConfigDict
) -> Iterable[tuple[str, Any]]:
  if isinstance(cfg, dict):
    return cfg.items()
  elif isinstance(cfg, ml_collections.ConfigDict):
    return cfg.items(preserve_field_references=True)
  else:
    raise TypeError(f'Invalid config dict: {cfg}')


_Field = ml_collections.FieldReference | ConfigDict
