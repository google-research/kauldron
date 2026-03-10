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

from __future__ import annotations

from etils import epath
from kauldron.contrib.train import npz_writer
import numpy as np
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
  return epath.Path(tmp_path)


def test_filter_arrays_no_patterns():
  values = {
      "summaries/embedding": np.ones((4, 8)),
      "summaries/logits": np.zeros((4, 10)),
      "summaries/text": "hello",
  }
  result = npz_writer._filter_arrays(values, patterns=None)
  assert set(result.keys()) == {"summaries/embedding", "summaries/logits"}


def test_filter_arrays_with_patterns():
  values = {
      "summaries/embedding": np.ones((4, 8)),
      "summaries/logits": np.zeros((4, 10)),
      "summaries/other": np.zeros((2,)),
  }
  result = npz_writer._filter_arrays(values, patterns=["logits"])
  assert set(result.keys()) == {"summaries/logits"}


def test_filter_arrays_regex_pattern():
  values = {
      "summaries/layer0/embedding": np.ones((4, 8)),
      "summaries/layer1/embedding": np.zeros((4, 8)),
      "summaries/logits": np.zeros((4, 10)),
  }
  result = npz_writer._filter_arrays(values, patterns=[r"layer\d+/embedding"])
  assert set(result.keys()) == {
      "summaries/layer0/embedding",
      "summaries/layer1/embedding",
  }


def test_filter_arrays_skips_non_arrays():
  values = {
      "summaries/arr": np.ones((2,)),
      "summaries/text": "not an array",
      "summaries/num": 42,
  }
  result = npz_writer._filter_arrays(values, patterns=None)
  assert set(result.keys()) == {"summaries/arr"}


def test_scalars_as_arrays():
  scalars = {"losses/total": 0.5, "metrics/accuracy": 0.9}
  result = npz_writer._scalars_as_arrays(scalars)
  assert set(result.keys()) == {"losses/total", "metrics/accuracy"}
  np.testing.assert_allclose(result["losses/total"], 0.5)
  np.testing.assert_allclose(result["metrics/accuracy"], 0.9)


def test_save_npz(tmp_dir):  # pylint: disable=redefined-outer-name
  writer = npz_writer.NpzWriter(
      output_dir=tmp_dir / "out",
      collection="train",
  )
  arrays = {
      "summaries/embedding": np.ones((4, 8)),
      "summaries/logits": np.zeros((4, 10)),
  }
  writer._save_npz(step=100, arrays=arrays)

  path = tmp_dir / "out" / "000000100.npz"
  assert path.exists()

  loaded = np.load(str(path))
  np.testing.assert_array_equal(loaded["summaries/embedding"], np.ones((4, 8)))
  np.testing.assert_array_equal(loaded["summaries/logits"], np.zeros((4, 10)))


def test_get_output_dir_custom(tmp_dir):  # pylint: disable=redefined-outer-name
  writer = npz_writer.NpzWriter(
      output_dir=tmp_dir / "custom",
      collection="train",
  )
  assert writer._get_output_dir() == tmp_dir / "custom"


def test_get_output_dir_default(tmp_dir):  # pylint: disable=redefined-outer-name
  writer = npz_writer.NpzWriter(
      workdir=tmp_dir / "workdir",
      collection="train",
  )
  expected = tmp_dir / "workdir" / "array_dumps" / "train"
  assert writer._get_output_dir() == expected
