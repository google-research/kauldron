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
import inspect
from typing import Any, Optional

from etils import enp
from etils import epy
from etils import etree
from flax import linen as nn
import jax
from kauldron import core
from kauldron import data
from kauldron import konfig
from kauldron import train
from kauldron.data import utils as data_utils
from kauldron.utils import pd_utils
import ml_collections
import numpy as np
import pandas as pd


def _get_source_link(cls) -> str:
  path = inspect.getfile(cls)
  lineno = inspect.getsourcelines(cls)[1]
  return f"file://{path}#l={lineno}"


def _convert_to_array_spec(x: Any) -> Any:
  if enp.ArraySpec.is_array(x):
    return enp.ArraySpec.from_array(x)
  elif isinstance(x, nn.summary._ArrayRepresentation):  # pylint: disable=protected-access
    return enp.ArraySpec(x.shape, x.dtype)
  elif isinstance(x, nn.summary._ObjectRepresentation):  # pylint: disable=protected-access
    return x.obj
  else:
    return x


def _format_module_config(cfg: Optional[Any]) -> str:
  """Return html span with emoji and tooltip containing abbreviated config."""
  if not cfg:
    return ""

  def _abbrev(c):
    if isinstance(c, ml_collections.ConfigDict):
      qn = getattr(c, "__qualname__")
      if qn is not None:
        return konfig.ConfigDict({"__qualname__": qn, 0: ...})
      else:
        return c
    return c

  cfg_abbrev = konfig.ConfigDict({k: _abbrev(v) for k, v in cfg.items()})
  return f'<span title="{cfg_abbrev}">📄</span>'


def _format_module_path(path: tuple[str, ...]) -> str:
  if len(path) > 1:
    prefix = f'<span style="color: gray">{".".join(path[:-1])}.</span>'
    return prefix + path[-1]
  else:
    return ".".join(path)


def _nbsp(s):
  """Return a repr of s with all spaces replaced by non-breaking ones."""
  return repr(s).replace(" ", "\xa0")


def _format_module(module_type: type[Any]) -> str:
  module_name = module_type.__name__
  module_path = f"{module_type.__module__}.{module_name}"
  module_src = _get_source_link(module_type)
  return f'<a href="{module_src}" title="{module_path}">{module_name}</a>'


def _format_annotation(annotation: Any) -> str:
  if annotations is None:
    return ""
  else:
    return str(annotation).removeprefix("typing.")


def _get_args(
    module_type: type[Any],
    method_name: str,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], str]:
  """Return inputs bound to method arguments along with annotations."""
  method = getattr(module_type, method_name)
  sig = inspect.signature(method)
  if not isinstance(inputs, dict):
    ba = sig.bind("self", _convert_to_array_spec(inputs))
  else:
    ba = sig.bind("self", **jax.tree_map(_convert_to_array_spec, inputs))
  args = ba.arguments
  del args["self"]

  input_ann = {
      k: _format_annotation(method.__annotations__.get(k)) for k in args
  }
  return_ann = _format_annotation(sig.return_annotation)

  return args, input_ann, return_ann


def _format_inputs(args, input_ann) -> str:
  return "<br>".join(
      f'<span title="{k}: {input_ann[k]}"><b>{k}</b>: {_nbsp(args[k])}</span>'
      for k in args
  )


def _format_outputs(outputs, return_ann) -> str:
  """Return a html output string for the outputs of a module."""
  if not isinstance(outputs, dict):
    return (
        "<span"
        f' title="{return_ann}">{_nbsp(_convert_to_array_spec(outputs))}</span>'
    )
  else:
    outputs = jax.tree_map(_convert_to_array_spec, outputs)
    flat_tree = core.tree_flatten_with_path(outputs)
    outputs = "<br>".join(
        f"<b>{k}</b>: {_nbsp(v)}" for k, v in flat_tree.items()
    )
    outputs = f'<span title="{return_ann}">{outputs}</span>'
  return outputs


def _format_param_shapes(module_variables) -> str:
  params = module_variables.get("params", {})
  if not params:
    return ""
  tree = jax.tree_map(_convert_to_array_spec, params)
  flat_tree = core.tree_flatten_with_path(tree)
  return "<br>".join(f"<b>{k}</b>: {_nbsp(v)}" for k, v in flat_tree.items())


def _get_num_params(module_variables) -> int:
  params = module_variables.get("params", {})

  def add_num_params(x, y) -> int:
    return x + np.prod(y.shape)

  num_params = jax.tree_util.tree_reduce(add_num_params, params, initializer=0)
  return num_params


def _get_cumulative_params(path: tuple[str, ...], table) -> int:
  path = ".".join(path)
  return sum(
      _get_num_params(row.module_variables)
      for row in table
      if ".".join(row.path).startswith(path)
  )


def _get_styled_df(table, model_config: konfig.ConfigDict) -> pd.DataFrame:
  """Return a styled pd.DataFrame for the model-overview table in colab."""
  df_rows = []
  for row in table:
    args, input_ann, return_ann = _get_args(
        row.module_type, row.method, row.inputs
    )
    m_config = core.get_by_path(model_config, ".".join(row.path))
    df_rows.append({
        "Cfg": _format_module_config(m_config),
        "Path": _format_module_path(row.path),
        "Module": _format_module(row.module_type),
        "Inputs": _format_inputs(args, input_ann),
        "Outputs": _format_outputs(row.outputs, return_ann),
        "Parameter Shapes": _format_param_shapes(row.module_variables),
        "Own Params": _get_num_params(row.module_variables),
        "Total Params": _get_cumulative_params(row.path, table),
    })

  df = pd_utils.StyledDataFrame(df_rows)
  df.current_style.set_properties(**{"text-align": "left"})
  df.current_style.set_table_styles(
      [dict(selector="th", props=[("text-align", "left")])]
  )
  df.current_style.format({
      "Own Params": lambda x: f"{x:,}" if x else "",
      "Total Params": lambda x: f"{x:,}" if x else "",
  })
  # Set alternating row backgrounds to semi-transparent grey
  df.current_style.set_table_styles([
      {
          "selector": "tbody tr:nth-child(even)",
          "props": [("background-color", "#8884")],
      },
      {
          "selector": "tbody tr:hover",
          "props": [("background-color", "#44f4")],
      },
  ])

  return df


def _get_summary_table(
    model: nn.Module, ds: data.TFDataPipeline
) -> nn.summary.Table:
  """Return model overview as a `nn.summary.Table`."""
  m_batch = data_utils.mock_batch_from_elem_spec(ds.element_spec)
  model_args, model_kwargs = data_utils.get_model_inputs(
      model, {"batch": m_batch, "step": 0}
  )
  table_fn = nn.summary._get_module_table(  # pylint: disable=protected-access
      model, depth=None, show_repeated=False
  )
  table = table_fn(
      {"params": jax.random.PRNGKey(0), "default": jax.random.PRNGKey(0)},
      *model_args,
      **model_kwargs,
  )
  return table


def json_spec_like(obj) -> Any:
  """Convert `etree.spec_like` output to json and displays it in colab form."""
  from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  spec = etree.spec_like(obj)

  def _to_json(spec):
    if isinstance(spec, Mapping):
      return {k: _to_json(v) for k, v in spec.items()}
    elif isinstance(spec, list):
      return [_to_json(v) for v in spec]
    elif isinstance(spec, tuple):
      return tuple(_to_json(v) for v in spec)
    elif isinstance(spec, (int, float, bool, type(None))):
      return spec
    else:
      return str(spec)

  json_spec = _to_json(spec)
  if epy.is_notebook():
    ecolab.json(json_spec)
  return json_spec


def eval_context_shape(model, losses, metrics, summaries, elem_spec):
  """Shape evaluate the model (fast) and return the shapes of the context."""
  m_rngs = {"default": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(0)}
  mwa = train.train_step.ModelWithAux(
      model=model, losses=losses, metrics=metrics, summaries=summaries
  )
  params = jax.eval_shape(mwa.init, init_rngs=m_rngs, elem_spec=elem_spec)
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


def get_colab_model_overview(
    model: nn.Module,
    train_ds: data.TFDataPipeline,
    model_config: konfig.ConfigDict,
) -> pd.DataFrame:
  """Return `pd.DataFrame` for displaying the model params, inputs,..."""
  table = _get_summary_table(model, train_ds)
  return _get_styled_df(table, model_config)
