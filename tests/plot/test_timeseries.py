# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
import os
import pytest
import sys

from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.plot.monitors import Plot
from tasmania.python.plot.trackers import TimeSeries


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_timeseries".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_datapoint(isentropic_data):
    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # make sure the folder tests/baseline_images/test_timeseries does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_datapoint_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    # indices identifying the grid point to visualize
    x, y, z = int(grid.nx / 2), int(grid.ny / 2), -1

    # drawer properties
    drawer_properties = {
        "linecolor": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
        "marker": "o",
    }

    # instantiate the drawer
    drawer = TimeSeries(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        z=z,
        time_mode="elapsed",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {"fontsize": 16, "figsize": (7, 8), "tight_layout": False}
    axes_properties = {
        "fontsize": 16,
        "title_center": "$x$ = {} km, $y$ = {} km, $\\theta$ = {} K".format(
            grid.x.to_units("km").values[x],
            grid.y.to_units("km").values[y],
            grid.z.to_units("K").values[z],
        ),
        "x_label": "Elapsed time [s]",
        "y_label": "$x$-velocity [m s$^{-1}$]",
        "grid_on": True,
    }

    # instantiate the monitor
    monitor = Plot(
        drawer,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    for state in states[:-1]:
        monitor.store(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


class MaxVelocity(DiagnosticComponent):
    def __init__(self, domain, grid_type):
        super().__init__(domain, grid_type)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
        return {"x_velocity_at_u_locations": {"dims": dims, "units": "km hr^-1"}}

    @property
    def diagnostic_properties(self):
        dims = ("scalar", "scalar", "scalar")
        return {"max_x_velocity_at_u_locations": {"dims": dims, "units": "km hr^-1"}}

    def array_call(self, state):
        val = np.max(state["x_velocity_at_u_locations"])
        return {
            "max_x_velocity_at_u_locations": np.array(val)[
                np.newaxis, np.newaxis, np.newaxis
            ]
        }


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_diagnostic(isentropic_data):
    # field to plot
    field_name = "max_x_velocity_at_u_locations"
    field_units = "km hr^-1"

    # make sure the folder tests/baseline_images/test_timeseries does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_diagnostic_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    # drawer properties
    drawer_properties = {
        "linecolor": "green",
        "linestyle": None,
        "linewidth": 1.5,
        "marker": "^",
        "markersize": 2,
    }

    # instantiate the drawer
    drawer = TimeSeries(
        grid,
        field_name,
        field_units,
        time_mode="elapsed",
        time_units="hr",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {"fontsize": 16, "figsize": (7, 8), "tight_layout": True}
    axes_properties = {
        "fontsize": 16,
        "x_label": "Elapsed time [hr]",
        "y_label": "Maximum $x$-velocity [km h$^{-1}$]",
        "grid_on": True,
        "grid_properties": {"linestyle": ":"},
    }

    # instantiate the monitor
    monitor = Plot(
        drawer,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # instantiate the diagnostic
    mv = MaxVelocity(domain, grid_type)

    # plot
    for state in states[:-1]:
        state.update(mv(state))
        monitor.store(state)
    states[-1].update(mv(states[-1]))
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
