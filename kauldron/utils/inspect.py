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
import dataclasses
import inspect
from typing import Any, Optional

from etils import enp
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
from kauldron import data
from kauldron import konfig
from kauldron import kontext
from kauldron import random as kd_random
from kauldron import train
from kauldron.data import utils as data_utils
from kauldron.typing import Float, UInt8  # pylint: disable=g-multiple-import
from kauldron.utils import pd_utils
import mediapy as media
import ml_collections
import numpy as np
import pandas as pd

# API import `kd.inspect.Profiler`
# pylint: disable=unused-import,g-bad-import-order,g-importing-member
from kauldron.utils.plotting import plot_schedules
from kauldron.utils.profile_utils import Profiler
from kauldron.utils.graphviz_utils import get_connection_graph
# pylint: enable=unused-import,g-bad-import-order,g-importing-member

_Example = Any


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
  if not isinstance(cfg, konfig.ConfigDict):
    return ""

  def _abbrev(c):
    if isinstance(c, ml_collections.ConfigDict):
      qn = getattr(c, "__qualname__", None)
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


def undo_process_input(inputs):
  # need to undo the `flax.linen.summary.process_inputs` function:
  # Note that contains some uncertainty, so we yield all different possibilities
  # in descending order of specificity. That way we can try to bind them to
  # the function signature and see what sticks.
  match inputs:
    case dict(kwargs):
      # single dictionary: could have come either from kwargs
      yield (), kwargs
      # or single (dict-type) positional argument
      yield (inputs,), {}
    case (*args, dict(kwargs)):
      # this form could have either come from *args, **kwargs
      yield args, kwargs
      # or from just args where the last arg happened to be a dict
      yield inputs, {}
      # or from single (tuple-typed) positional argument
      yield (inputs,), {}
    case tuple(args):
      # either came from *args (and no kwargs)
      yield args, {}
      # or from a single (tuple-typed) positional arg
      yield (inputs,), {}
    case _:
      # this can only come from single positional arg
      yield (inputs,), {}


def _get_args(
    module_type: type[Any],
    method_name: str,
    inputs: Mapping[str, Any] | tuple[Any, ...] | Any,
) -> tuple[dict[str, Any], dict[str, str], str]:
  """Return inputs bound to method arguments along with annotations."""
  method = getattr(module_type, method_name)
  sig = inspect.signature(method)

  inputs = jax.tree_map(_convert_to_array_spec, inputs)
  for args, kwargs in undo_process_input(inputs):
    try:
      ba = sig.bind("self", *args, **kwargs)
      args = ba.arguments
      del args["self"]
      break
    except TypeError:
      pass  # try the next interpretation
  else:
    print(f"Error: Failed to bind inputs to {method_name} of {module_type}.")
    print("Inputs:", inputs)
    args = {"ERROR": "Failed to bind inputs"}

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


def _format_as_bullet_points(items):
  bps = "\n".join([f"<li>{s}</li>" for s in items])
  return "<ul>" + bps + "</ul>"


def _format_as_numbered_list(items):
  numlist = "\n".join([f"<li>{s}</li>" for s in items])
  return "<ol>" + numlist + "</ol>"


def _format_nested_args(args):
  if isinstance(args, dict):
    return _format_as_bullet_points(
        [f"<b>{k}</b>: {_format_nested_args(v)}" for k, v in args.items()]
    )
  elif isinstance(args, tuple):
    return _format_as_numbered_list([_format_nested_args(v) for v in args])
  elif isinstance(args, set):
    return _format_as_bullet_points([_format_nested_args(v) for v in args])
  else:
    return _nbsp(_convert_to_array_spec(args))


def _format_outputs(outputs, return_ann) -> str:
  """Return a html output string for the outputs of a module."""
  return f'<span title="{return_ann}">{_format_nested_args(outputs)}</span>'


def _format_param_shapes(module_variables) -> str:
  params = module_variables.get("params", {})
  if not params:
    return ""
  tree = jax.tree_map(_convert_to_array_spec, params)
  flat_tree = kontext.flatten_with_path(tree)
  return "<br>".join(f"<b>{k}</b>: {_nbsp(v)}" for k, v in flat_tree.items())


def _get_num_params(module_variables) -> int:
  params = module_variables.get("params", {})

  def add_num_params(x, y) -> int:
    return x + np.prod(y.shape)

  num_params = jax.tree_util.tree_reduce(add_num_params, params, initializer=0)
  return num_params


def _get_cumulative_params(path: tuple[str, ...], table) -> int:
  path = ".".join(path)

  def is_part_of_path(row_path) -> bool:
    str_path = ".".join(row_path)
    return str_path == path or str_path.startswith(path + ".")

  return sum(
      _get_num_params(row.module_variables)
      for row in table
      if is_part_of_path(row.path)
  )


def _get_styled_df(
    table: nn.summary.Table, model_config: konfig.ConfigDict
) -> pd.DataFrame:
  """Return a styled pd.DataFrame for the model-overview table in colab."""
  df_rows = []
  for row in table:
    args, input_ann, return_ann = _get_args(
        row.module_type, row.method, row.inputs
    )
    # It's still possible that the module path conflict with an attribute, like
    # * In the config: `model = MyModel(some_value=Config())`
    # * Inside `MyModel.__call__`: `x = nn.Dense(name='some_value')(x)`
    # Here `get_by_path(, 'some_value')` will extract the `Config()`, instead
    # of returning `None`.
    m_config = kontext.get_by_path(
        model_config, ".".join(row.path), default=None
    )
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
    model: nn.Module,
    ds: data.Pipeline,
    rngs: dict[str, kd_random.PRNGKey],
) -> nn.summary.Table:
  """Return model overview as a `nn.summary.Table`."""
  m_batch = data_utils.mock_batch_from_elem_spec(ds.element_spec)
  model_args, model_kwargs = data_utils.get_model_inputs(
      model, {"batch": m_batch, "step": 0}
  )
  table_fn = nn.summary._get_module_table(  # pylint: disable=protected-access
      model,
      depth=None,
      show_repeated=False,
      compute_flops=False,
      compute_vjp_flops=False,
  )

  table = table_fn(
      rngs,
      *model_args,
      is_training_property=True,
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


def get_colab_model_overview(
    model: nn.Module,
    train_ds: data.Pipeline,
    model_config: konfig.ConfigDict,
    rngs: dict[str, kd_random.PRNGKey],
) -> pd.DataFrame:
  """Return `pd.DataFrame` for displaying the model params, inputs,..."""
  table = _get_summary_table(model, train_ds, rngs)
  return _get_styled_df(table, model_config)


def get_batch_stats(batch: _Example) -> pd.DataFrame:
  """Return `pd.DataFrame` containing the batch stats."""
  # TODO(epot):
  # * Supports string too
  return pd.DataFrame(
      [
          {
              "Name": f"batch.{k}",
              "Dtype": v.dtype,
              "Shape": v.shape,
              "Min": np.min(v),
              "Max": np.max(v),
              "Mean": np.mean(v),
              "StdDev": np.std(v),
          }
          for k, v in kontext.flatten_with_path(batch).items()
      ]
  )


def plot_batch(batch: _Example) -> None:
  """Display batch images."""
  # pylint: disable=invalid-name
  ImagesGrayscale = Float["b h w 1"] | UInt8["b h w 1"]
  ImagesRGB = Float["b h w 3"] | UInt8["b h w 3"]
  ImagesRGBA = Float["b h w 4"] | UInt8["b h w 4"]
  Images = ImagesGrayscale | ImagesRGB | ImagesRGBA

  VideosRGB = Float["b t h w 3"] | UInt8["b t h w 3"]
  # pylint: enable=invalid-name

  for k, v in batch.items():
    if isinstance(v, Images):
      _, height, _, _ = v.shape
      height = _normalize_height(height)
      media.show_images(
          v[:8],
          ylabel=f'<span style="font-size: 20;">batch.{k}</span>',
          height=height,
      )
    elif isinstance(v, VideosRGB):
      # Dynamically compute the frame-rate, capped at 25 FPS
      _, num_frames, height, _, _ = v.shape
      height = _normalize_height(height)
      fps = min(num_frames // 5, 25.0)

      media.show_videos(
          v[:8],
          ylabel=f'<span style="font-size: 20;">batch.{k}</span>',
          fps=fps,
          height=height,
      )


def _normalize_height(
    height: int,
    *,
    min_height: int = 100,
    max_height: int = 250,
) -> int:
  """Truncate height to be within the range."""
  height = max(height, min_height)
  height = min(height, max_height)
  return height


def plot_context(config: train.Config) -> None:
  """Display the context structure."""
  context = config.context_specs
  context = {
      f.name: getattr(context, f.name) for f in dataclasses.fields(context)
  }
  context["grads"] = "[same as params]"
  context["updates"] = "[same as params]"
  json_spec_like(context)
