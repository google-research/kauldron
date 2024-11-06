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

"""Display some visualization for the trainer."""

from __future__ import annotations

from etils import epy
from kauldron.inspect import graphviz_utils
from kauldron.inspect import inspect as inspect_lib
from kauldron.inspect import plotting
from kauldron.train import trainer_lib

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from etils import ecolab
  import IPython.display
  # pylint: enable=g-import-not-at-top  # pytype: enable=import-error


def show_trainer_info(
    trainer: trainer_lib.Trainer,
    *,
    print_cfg: bool,
    inspect_dataset: bool,
    inspect_params: bool,
    inspect_sharding: bool,
    inspect_connected: bool,
    profile_statix: bool,
):
  """Display various plot on the trainer."""
  cfg = trainer.raw_cfg

  if print_cfg:
    with ecolab.collapse("Config (modified)"):
      ecolab.disp(cfg)

  if inspect_dataset:
    batch = next(iter(trainer.train_ds))

    with ecolab.collapse("Batch statistics"):
      ecolab.disp(inspect_lib.get_batch_stats(batch))

    with ecolab.collapse("Batch images"):
      inspect_lib.plot_batch(batch)

  if inspect_params:

    # Model Overview
    model_overview = inspect_lib.get_colab_model_overview(
        model=trainer.model,
        model_config=None if cfg is None else cfg.model,
        train_ds=trainer.train_ds,
        ds_sharding=trainer.sharding.ds,
        rngs=trainer.rng_streams.init_rngs(),
    )
    total_params = model_overview["Own Params"].sum()
    with ecolab.collapse(f"Model Overview (#Params: {total_params:,})"):
      ecolab.disp(model_overview)

    with ecolab.collapse("Structure of Context:"):
      inspect_lib.plot_context(trainer)

    if trainer.schedules:
      with ecolab.collapse("Schedules"):
        fig = plotting.plot_schedules(
            trainer.schedules,
            num_steps=trainer.num_train_steps,
        )
        # TODO(b/299308317): Remove `IPython.display.HTML`
        ecolab.disp(IPython.display.HTML(fig.to_html()))

  if inspect_connected:
    with ecolab.collapse("Connected components:"):
      ecolab.disp(graphviz_utils.get_connection_graph(trainer))

  if inspect_sharding:
    with ecolab.collapse("Sharding"):
      inspect_lib.plot_sharding(trainer)
