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

"""Model exporters are used to serialize the model computations.

(as opposed to the parameters which are saved in the checkpoint)
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
from typing import Any, Literal, Optional, Sequence, TYPE_CHECKING

from etils import epath
import flax.linen as nn
import jax
from jax import export
from jax.experimental import checkify
from kauldron.data import utils as data_utils
from kauldron.utils import config_util
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

if TYPE_CHECKING:
  # pylint: disable=g-bad-import-order
  from kauldron.train import rngs_lib
  from kauldron.train import train_step


_DEFAULT_EXPORTED_NAME = 'train_model'


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ModelExporter(abc.ABC, config_util.UpdateFromRootCfg):
  """Called at the beginning of training to export the model.

  Attributes:
    name: The name of the exported model.
    workdir: The workdir to export the model to.
    rng_streams: The rng streams to use for exporting.
  """

  name: str = _DEFAULT_EXPORTED_NAME

  workdir: epath.Path = config_util.ROOT_CFG_REF.workdir
  rng_streams: rngs_lib.RngStreams = config_util.ROOT_CFG_REF.rng_streams

  __root_cfg_fields_to_recurse__ = ('rng_streams',)

  def get_rngs(self, is_training: bool, step: int = 0) -> rngs_lib.RngStreams:
    self._assert_root_cfg_resolved()
    if is_training:
      return self.rng_streams.train_rngs(step)
    else:
      return self.rng_streams.eval_rngs(step)

  @abc.abstractmethod
  def export(
      self,
      *,
      model: nn.Module,
      state: train_step.TrainState,
      element_spec: Any,
      is_training: bool,
  ) -> None:
    """Exports the model to a serialized format.

    Args:
      model: The model to export.
      state: The trainer state containing the model parameters.
      element_spec: The element spec of the dataset.
      is_training: Whether the model is to be called in training mode.
    """


class NoopExporter(ModelExporter):
  """Noop exporter."""

  def export(
      self,
      *,
      model: nn.Module,
      state: train_step.TrainState,
      element_spec: Any,
      is_training: bool,
  ) -> None:
    pass


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class JaxModelExporter(ModelExporter):
  """Exports a model to a jax serialized model.

  Attributes:
    batch_specs: Specifies which dimensions should be treated as variable.
      Default is 'b, ...', which means that the first dimension of all batch
      elements is variable size, and all other dimensions are assumed fixed
      size. `batch_specs` can be a string or a pytree of strings (matching the
      structure of the batch).
    model_method: The name of the model method to call (if None, defaults to
      `__call__`).
    platforms: The list of platforms to export the model for. Options are 'cpu',
      'tpu', 'cuda', and 'rocm'.
    path_template: The path template to use for exporting. Can contain
      {workdir}, {name}, and {train_or_eval} placeholders.
  """

  batch_specs: Any = 'b, ...'  # prefix pytree of strings
  model_method: Optional[str] = None

  platforms: Sequence[Literal['cpu', 'tpu', 'cuda', 'rocm']] = ('cpu', 'tpu')
  vjp_order: int = 1

  path_template: str = '{workdir}/{name}.jax_exported'

  ds_sharding: sharding.ShardingTree = config_util.ROOT_CFG_REF.sharding.batch  # pyrefly: ignore[not-a-type]

  def export(
      self,
      *,
      model: nn.Module,
      state: train_step.TrainState,
      element_spec: Any,
      is_training: bool,
  ) -> None:
    self._assert_root_cfg_resolved()
    if not status.is_lead_host:
      return  # Only the lead host should export the model.

    mock_batch = data_utils.mock_batch_from_elem_spec(
        element_spec, self.ds_sharding
    )
    symb_batch_spec = export.symbolic_args_specs(mock_batch, self.batch_specs)
    from kauldron.train import context as context_lib  # pylint: disable=g-import-not-at-top
    context = context_lib.Context.from_state_and_batch(
        state=state, batch=symb_batch_spec
    )
    args, kwargs = data_utils.get_model_inputs(model, context)
    assert not args

    forward_fn = _create_dynamic_forward_fn(
        model=model,
        method=self.model_method,
        is_training=is_training,
        rngs=self.get_rngs(is_training),
        kwarg_names=list(kwargs.keys()),
    )

    # TODO(klausg): maybe also export a version with a single device sharding?
    # https://docs.jax.dev/en/latest/export/export.html#device-polymorphic-export
    exported = export.export(jax.jit(forward_fn), platforms=self.platforms)(
        params=state.params,
        collections=state.collections,
        key=jax.random.PRNGKey(0),
        **kwargs,
    )
    blob = exported.serialize(vjp_order=self.vjp_order)

    # Write to the specified path.
    path = epath.Path(
        self.path_template.format(
            workdir=self.workdir,
            name=self.name,
        )
    )
    path.write_bytes(blob)


def _create_dynamic_forward_fn(
    *,
    model: nn.Module,
    method: str | None = None,
    is_training: bool = False,
    rngs: rngs_lib.RngStreams,
    kwarg_names: Sequence[str],
):
  """Creates a forward function for the model with a custom signature."""

  def _forward(*, params, collections, key, **kwargs):
    # Fold in the key to the individual rng streams
    # (which are treated as constants)
    bits = jax.random.bits(key)
    new_rngs = jax.tree.map(lambda r: jax.random.fold_in(r, bits), rngs)

    variables = {'params': params} | collections
    model_apply_checkified = checkify.checkify(model.apply)
    error, (preds, out_collections) = model_apply_checkified(
        variables,
        rngs=new_rngs,
        mutable=True,
        capture_intermediates=True,
        is_training_property=is_training,
        method=method,
        **kwargs,
    )
    del error  # ignore checkify errors for export

    return {'preds': preds, 'interms': out_collections['intermediates']}

  # Create and set a dynamic signature for the function.
  parameters = [
      inspect.Parameter('params', inspect.Parameter.KEYWORD_ONLY),
      inspect.Parameter('collections', inspect.Parameter.KEYWORD_ONLY),
      inspect.Parameter('key', inspect.Parameter.KEYWORD_ONLY),
  ] + [
      inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
      for name in kwarg_names
  ]
  new_signature = inspect.Signature(parameters)
  _forward.__signature__ = new_signature  # pyrefly: ignore[missing-attribute]

  # Set the name of the function
  _forward.__name__ = method or '__call__'

  return _forward
