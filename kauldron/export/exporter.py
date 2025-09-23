# Copyright 2025 The kauldron Authors.
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

import abc
import dataclasses
from typing import Any, Optional, Sequence
from etils import epath
import flax.linen as nn
import jax
from jax import export
from jax.experimental import checkify
from kauldron import train
from kauldron.data import utils as data_utils
from kauldron.typing import PRNGKey
from kauldron.utils import config_util
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

_DEFAULT_EXPORTED_NAME = 'exported_model'


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
  rng_streams: train.RngStreams = config_util.ROOT_CFG_REF.rng_streams

  __root_cfg_fields_to_recurse__ = ('rng_streams',)

  def get_rngs(self, is_training: bool, step: int = 0) -> train.RngStreams:
    self._assert_root_cfg_resolved()
    if is_training:
      return self.rng_streams.train_rngs(step)
    else:
      return self.rng_streams.eval_rngs(step)

  @abc.abstractmethod
  def export(
      self,
      model: nn.Module,
      state: train.TrainState,
      element_spec: Any,
      is_training: bool,
      ds_sharding: sharding.ShardingTree,
  ) -> None:
    """Exports the model to a serialized format.

    Args:
      model: The model to export.
      state: The trainer state containing the model parameters.
      element_spec: The element spec of the dataset.
      is_training: Whether the model is to be called in training mode.
      ds_sharding: The sharding of the dataset. Usually `trainer.sharding.ds`.
    """


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

  platforms: Sequence[str] = ('cpu', 'tpu')  # "cuda", "rocm"

  path_template: str = '{workdir}/{name}_{train_or_eval}.jax_exported'

  def export(
      self,
      model: nn.Module,
      state: train.TrainState,
      element_spec: Any,
      is_training: bool,
      ds_sharding: sharding.ShardingTree,
  ) -> None:
    self._assert_root_cfg_resolved()
    if not status.is_lead_host:
      return  # Only the lead host should export the model.

    path = epath.Path(
        self.path_template.format(
            workdir=self.workdir,
            name=self.name,
            train_or_eval='train' if is_training else 'eval',
        )
    )
    mock_batch = data_utils.mock_batch_from_elem_spec(element_spec, ds_sharding)

    blob = serialize_model(
        model,
        state=state.replace(opt_state=None),
        batch=mock_batch,
        rngs=self.get_rngs(is_training),
        is_training=True,
        model_method=self.model_method,
        platforms=self.platforms,
        specs=self.batch_specs,
    )
    path.write_bytes(blob)


def serialize_model(
    model: nn.Module,
    *,
    state: train.TrainState,
    batch,
    rngs,
    is_training: bool = False,
    model_method: str | None = None,
    platforms: Sequence[str] = ('cpu', 'tpu'),  # "cuda", "rocm"
    vjp_order: int = 1,
    specs: str = 'b, ...',
) -> bytes:
  """Serializes a Kauldron model's forward pass using `jax.export`.

  Args:
    model: The linen module to serialize.
    state: The Kauldron training state containing model parameters.
    batch: A batch (or mock batch) with the correct structure.
    rngs: JAX PRNG keys for the forward pass.
    is_training: Whether to export in training or evaluation mode.
    model_method: The name of the model method to call (if None, calls
      `__call__`).
    platforms: List of target platforms for export (available options are 'cpu',
      'tpu', 'cuda' and 'rocm').
    vjp_order: The VJP order for serialization. E.g. if vjp_order=1 (the
      default), the serialized model supports a single application of
      `jax.grad`.
    specs: Symbolic dimension specifications for batch elements. Default is
      'b,...', which means that the first dimension of all batch elements is
      variable size, and all other dimensions are assumed fixed size. `specs`
      can be a string or a pytree of strings.

  Returns:
    The serialized JAX exported model as bytes.
  """

  def forward(
      batch: Any,
      params: Any,
      rng_key: PRNGKey = jax.random.PRNGKey(0),
      step: int = 0,
  ) -> train.Context:
    forward_checkified = checkify.checkify(train.forward)
    # Construct initial context
    # TODO(klausg): support collections?
    train_state = train.TrainState(
        step=step, params=params, collections={}, opt_state=None
    )
    ctx = train.Context.from_state_and_batch(state=train_state, batch=batch)

    # Fold in the rng_key to the individual rng streams
    # (which are treated as constants)
    bits = jax.random.bits(rng_key)
    new_rngs = jax.tree.map(lambda r: jax.random.fold_in(r, bits), rngs)

    # Call the forward pass
    error, context = forward_checkified(
        model=model,
        context=ctx,
        rngs=new_rngs,
        is_training=is_training,
        method=model_method,
    )
    del error  # ignore checkify errors for export
    outputs = {'preds': context.preds, 'interms': context.interms}
    if context.collections is not None:
      outputs['collections'] = context.collections
    return outputs

  symb_batch_spec = export.symbolic_args_specs(batch, specs)

  exported = export.export(jax.jit(forward), platforms=platforms)(
      symb_batch_spec, state.params, jax.random.PRNGKey(0), step=0
  )
  return exported.serialize(vjp_order=vjp_order)
