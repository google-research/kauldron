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

"""Build utils."""

from __future__ import annotations

import datetime
import getpass

from etils import epy
from kauldron.utils.kdash import dashboard_utils
from kauldron.utils.kdash import plot_utils
from kauldron.utils.kdash import xm_utils
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member


def build_and_upload(
    dashboards: dashboard_utils.DashboardsBase,
    *,
    ctx: plot_utils.BuildContext | None = None,
) -> None:
  """Create the dashboards.

  * On XManager: Add the articfacts for each dashboard.
  * On Colab: Print the links.

  Args:
    dashboards: The dashboards to build.
    ctx: Build context.
  """
  # TODO(epot): Remove `status.wid == 1`. Might be a case where each
  # work-units have different dashboards (but likely not super frequent,
  # so might be fine for now).
  if status.on_xmanager and not (status.is_lead_host and status.wid == 1):
    return  # Only lead host can create dashboards.

  ctx = ctx or plot_utils.BuildContext()

  dashboards = dashboards.normalize().build(ctx)
  if epy.is_test():  # Inside tests, do not actually create the dashboards.
    return
  urls = {
      # Remove the `/revisions/` so that flatboard updates are reflected.
      # TODO(epot): Is it a good idea?
      k: db.save_url(
          reader_permissions=fb.FlatboardDashboardPermissions.EVERYONE
      ).split('/revisions/')[0]
      for k, db in dashboards.items()
  }
  status.log('Dashboards:')
  for k, url in urls.items():
    status.log(f'* {k}: {url}')

  # Add the XManager atrifacts to setup the UI on XManager.
  # Only add the flatboard artifacts for the first work-unit due to a race
  # condition between work-units (b/351986366).
  if status.on_xmanager and status.wid == 1:
    for k, url in urls.items():
      xm_utils.add_flatboard_artifact(k, url)


def make_collection_path_prefix() -> str:
  """Returns the collection path prefix to use."""
  if status.on_xmanager:
    return f'/datatable/xid/{status.xid}/'
  else:
    curr_date = datetime.datetime.now()
    curr_date = curr_date.strftime('%Y%m%d%H%M%S')
    return f'/datatable/users/{_get_user()}/t{curr_date}.'


def _get_user() -> str:
  if epy.is_notebook():
    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return ecolab.getuser()
  else:
    return getpass.getuser()
