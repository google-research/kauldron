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

# %% [markdown]
# <!--
# pylint: disable=g-import-not-at-top
# pylint: disable=missing-module-docstring
# pylint: disable=pointless-statement
# pylint: disable=reimported
# pylint: disable=undefined-variable
# pylint: disable=unnecessary-semicolon
# pylint: disable=unused-import
# pylint: disable=used-before-assignment
# pylint: disable=wildcard-import
# -->
#
# # Kauldron Nerf - Minimal
#
# Minimal Colab template for experiment.

# %% Imports
from __future__ import annotations

from etils.lazy_imports import *

source = ''  # @param {type: "string"}

# Reload `etils` from head
with ecolab.adhoc(source, reload='etils'):
  from etils.lazy_imports import *

v3d.auto_plot_figs()  # Trigger import (before reloading)
# %%
# Clear dataset cache
globals().get('_colab_cache', {}).clear()

with ecolab.adhoc(source, reload='visu3d,dataclass_array'):
  import visu3d as v3d
  import dataclass_array as dca

v3d.auto_plot_figs()
# %%
with ecolab.adhoc(source, reload='kauldron', restrict_reload=False):
  from kauldron import kd
  from kauldron.projects.nerf import nerf

  from kauldron.projects.nerf.configs import base

cfg = base.get_config()
trainer = kd.konfig.resolve(cfg)

with ecolab.collapse('Trainer'):
  trainer;
# %%
plotly_cm = ecolab.adhoc()  # Set auto adhoc import for plotly
plotly_cm.__enter__()
# %%
