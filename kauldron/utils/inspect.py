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

"""Helpers to inspect kd models."""
from __future__ import annotations

from collections.abc import Mapping
import functools
import inspect
from typing import Any, Optional

from etils import ecolab
from etils import enp
from etils import etree
import flax.linen as nn
import jax
from kauldron import core
from kauldron import konfig
from kauldron import train
from kauldron.data import utils as data_utils
import ml_collections
import numpy as np
import pandas as pd


def get_source_link(cls) -> str:
  path = inspect.getfile(cls)
  lineno = inspect.getsourcelines(cls)[1]
  return f"file://{path}#l={lineno}"


def convert_to_array_spec(x: Any) -> Any:
  if hasattr(x, "shape") and hasattr(x, "dtype"):
    return enp.ArraySpec(x.shape, x.dtype)
  else:
    return x


def format_module_config(cfg: Optional[Any]) -> str:
  """Return html span with emoji and tooltip containing abbreviated config."""
  if not cfg:
    return ""

  def _abbrev(c):
    if isinstance(c, ml_collections.ConfigDict):
      qn = getattr(c, "__qualname__")
      if qn is not None:
        # return kd.konfig.configdict_base._normalize_qualname(qn) +"(...)"
        return konfig.configdict_base.ConfigDict({"__qualname__": qn, 0: ...})
      else:
        return c
    return c

  cfg_abbrev = konfig.configdict_base.ConfigDict(
      {k: _abbrev(v) for k, v in cfg.items()}
  )
  return f'<span title="{cfg_abbrev}">📄</span>'


def format_module_path(path: tuple[str, ...]) -> str:
  if len(path) > 1:
    prefix = f'<span style="color: gray">{".".join(path[:-1])}.</span>'
    return prefix + path[-1]
  else:
    return ".".join(path)


def format_module(module_type: type[Any]) -> str:
  module_name = module_type.__name__
  module_path = f"{module_type.__module__}.{module_name}"
  module_src = get_source_link(module_type)
  return f'<a href="{module_src}" title="{module_path}">{module_name}</a>'


def format_annotation(annotation: Any) -> str:
  if annotations is None:
    return ""
  else:
    return str(annotation).removeprefix("typing.")


def get_args(
    module_type: type[Any],
    method_name: str,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], str]:
  """Return inputs bound to method arguments along with annotations."""
  method = getattr(module_type, method_name)
  sig = inspect.signature(method)
  if not isinstance(inputs, dict):
    ba = sig.bind("self", convert_to_array_spec(inputs))
  else:
    ba = sig.bind("self", **jax.tree_map(convert_to_array_spec, inputs))
  args = ba.arguments
  del args["self"]

  input_ann = {
      k: format_annotation(method.__annotations__.get(k)) for k in args
  }
  return_ann = format_annotation(sig.return_annotation)

  return args, input_ann, return_ann


def format_inputs(args, input_ann) -> str:
  return "<br>".join(
      f'<span title="{k}: {input_ann[k]}"><b>{k}</b>: {args[k]}</span>'
      for k in args
  )


def format_outputs(outputs, return_ann) -> str:
  if not isinstance(outputs, dict):
    return f'<span title="{return_ann}">{convert_to_array_spec(outputs)}</span>'
  else:
    outputs = jax.tree_map(convert_to_array_spec, outputs)
    flat_tree = core.tree_flatten_with_path(outputs)
    outputs = "<br>".join(f"<b>{k}</b>: {v}" for k, v in flat_tree.items())
    outputs = f'<span title="{return_ann}">{outputs}</span>'
  return outputs


def format_param_shapes(module_variables) -> str:
  params = module_variables.get("params", {})
  if not params:
    return ""
  tree = jax.tree_map(convert_to_array_spec, params)
  flat_tree = core.tree_flatten_with_path(tree)
  return "<br>".join(f"<b>{k}</b>: {v}" for k, v in flat_tree.items())


def get_num_params(module_variables) -> int:
  params = module_variables.get("params", {})

  def add_num_params(x, y) -> int:
    return x + np.prod(y.shape)

  return jax.tree_util.tree_reduce(add_num_params, params, initializer=0)


def get_styled_df(table, model_config):
  """Return a styled pd.DataFrame for the model-overview table in colab."""
  df_rows = []
  for row in table:
    args, input_ann, return_ann = get_args(
        row.module_type, row.method, row.inputs
    )
    m_config = core.get_by_path(model_config, ".".join(row.path))
    df_rows.append({
        "Cfg": format_module_config(m_config),
        "Path": format_module_path(row.path),
        "Module": format_module(row.module_type),
        "Inputs": format_inputs(args, input_ann),
        "Outputs": format_outputs(row.outputs, return_ann),
        "Parameter Shapes": format_param_shapes(row.module_variables),
        "Num Params": get_num_params(row.module_variables),
    })

  df = pd.DataFrame(df_rows)
  df_styled = df.style.set_properties(**{"text-align": "left"})
  df_styled.set_table_styles(
      [dict(selector="th", props=[("text-align", "left")])]
  )
  df_styled.format({"Num Params": "{:,}"})
  return df, df_styled


def get_summary_table(model, ds):
  m_batch = data_utils.mock_batch_from_elem_spec(ds.element_spec)
  model_args, model_kwargs = data_utils.get_model_inputs(
      model, {"batch": m_batch, "step": 0}
  )
  table_fn = nn.summary._get_module_table(  # pylint: disable=protected-access
      model, depth=None, show_repeated=False
  )
  table = table_fn(
      {"params": jax.random.PRNGKey(0)}, *model_args, **model_kwargs
  )
  return table


@functools.cache
def is_notebook() -> bool:
  try:
    import IPython  # pylint: disable=g-import-not-at-top

    ipython = IPython.get_ipython()
    return ipython is not None
  except (ImportError, NameError, AttributeError):
    return False


def json_spec_like(obj) -> Any:
  """Convert `etree.spec_like` output to json and displays it in colab form."""
  spec = etree.spec_like(obj)

  def _to_json(spec):
    if isinstance(spec, Mapping):
      return {k: _to_json(v) for k, v in spec.items()}
    elif isinstance(spec, list):
      return [_to_json(v) for v in spec]
    elif isinstance(spec, tuple):
      return tuple(_to_json(v) for v in spec)
    elif isinstance(spec, (int, bool, type(None))):
      return spec
    else:
      return str(spec)

  json_spec = _to_json(spec)
  if is_notebook():
    ecolab.json(json_spec)
  return json_spec


def eval_context_shape(model, losses, metrics, summaries, elem_spec):
  """Shape evaluate the model (fast) and return the shapes of the context."""
  m_rngs = {"default": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(0)}
  mwa = train.train_step.ModelWithAux(
      model=model, losses=losses, metrics=metrics, summaries=summaries
  )
  params = jax.eval_shape(mwa.init, init_rng=m_rngs, elem_spec=elem_spec)
  m_batch = data_utils.mock_batch_from_elem_spec(elem_spec)
  loss, context = jax.eval_shape(
      mwa.forward,
      params=params,
      batch=m_batch,
      rngs=m_rngs,
      step=0,
      is_training=True,
  )
  return loss, context


def get_colab_model_overview(model, train_ds, model_config):
  table = get_summary_table(model, train_ds)
  _, df_styled = get_styled_df(table, model_config)
  return df_styled
