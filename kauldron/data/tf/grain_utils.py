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

"""Grain utils to abstract the meta features."""

import sys
from typing import Any

from etils import epy
from etils.etree import nest as etree  # pylint: disable=g-importing-member
import grain.tensorflow as grain
from kauldron import random
import tensorflow as tf


def _unpexected_example_structure(ex: Any) -> Exception:
  """Drop grain meta features."""
  spec = epy.pretty_repr(etree.spec_like(ex))
  return ValueError(f"Unexpected example structure: {spec}")


def split_grain_meta_features(
    features: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
  """Extract the non-grain features."""
  if not isinstance(features, dict) or grain.INDEX not in features:
    raise _unpexected_example_structure(features)

  # b/326590491: Grain meta features are inconsitent. Either:
  # * `{**meta, **X}`
  # * `{**meta, '_record': X}`
  if grain.RECORD in features:
    if any(not k.startswith("_") for k in features):
      raise _unpexected_example_structure(features)
    ex_features = features.pop(grain.RECORD)
    meta_features = features
  else:  # Dict and keys merged
    ex_features, meta_features = epy.splitby(
        features.items(),
        predicate=lambda key_and_value: key_and_value[0] in grain.META_FEATURES,
    )
    meta_features = dict(meta_features)
    ex_features = dict(ex_features)
  return meta_features, ex_features


def merge_grain_meta_features(meta_features, ex_features):
  if not isinstance(ex_features, dict):
    ex_features = {grain.RECORD: ex_features}
  return meta_features | ex_features


def maybe_add_grain_meta_features(
    ds,
    *,
    rng: random.PRNGKey,
) -> Any:
  """Add grain meta features."""
  # Dataset already has grain meta features.
  if isinstance(ds.element_spec, dict) and grain.INDEX in ds.element_spec:
    return ds

  # This should be deterministic as long as the `ds` is deterministic.
  sampler = grain.TfDefaultIndexSampler(
      num_records=sys.maxsize,  # Infinite iterator
      shuffle=False,
      seed=rng.fold_in("grain_metadata").as_seed(),
      shard_options=grain.ShardByJaxProcess(),
      num_epochs=None,
  )
  index_ds = sampler.get_index_dataset(grain.FirstIndex())
  ds = tf.data.Dataset.zip(index_ds, ds)
  ds = ds.map(merge_grain_meta_features)
  return ds
