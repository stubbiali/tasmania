# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
import os
import pytest
import sys

from tasmania.python.plot.monitors import Plot
from tasmania.python.plot.trackers import HovmollerDiagram


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_hovmoller".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_x(isentropic_data):
    # field to plot
    field_name = "horizontal_velocity"
    field_units = "km hr^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_x_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # indices identifying the line to visualize
    y, z = int(grid.ny / 2), -1

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 18,
        "cbar_ticks_step": 4,
        "cbar_ticks_pos": "center",
        "cbar_center": 15,
        "cbar_x_label": "Surface velocity [km h$^{-1}$]",
        "cbar_orientation": "horizontal",
    }

    # instantiate the drawer
    drawer = HovmollerDiagram(
        grid,
        field_name,
        field_units,
        y=y,
        z=z,
        axis_units="km",
        time_mode="elapsed",
        time_units="hr",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$y$ = {} km".format(grid.y.to_units("km").values[y]),
        "x_label": "$x$ [km]",
        "y_label": "Elapsed time [hr]",
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_z(isentropic_data):
    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_z_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # indices identifying the line to visualize
    x, y = int(grid.nx / 2), int(grid.ny / 2)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 18,
        "cbar_ticks_step": 4,
        "cbar_ticks_pos": "center",
        "cbar_center": 15,
        "cbar_orientation": "horizontal",
    }

    # instantiate the drawer
    drawer = HovmollerDiagram(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        axis_name="z",
        axis_units="K",
        time_mode="elapsed",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x$-velocity [m/s] at ($x$ = {} km, $y$ = {} km)".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "x_label": "Elapsed time [s]",
        "y_label": "$\\theta$ [K]",
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_pressure(isentropic_data):
    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_pressure_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # indices identifying the line to visualize
    x, y = int(grid.nx / 2), int(grid.ny / 2)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 18,
        "cbar_ticks_step": 4,
        "cbar_ticks_pos": "center",
        "cbar_center": 15,
        "cbar_orientation": "horizontal",
    }

    # instantiate the drawer
    drawer = HovmollerDiagram(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        axis_name="air_pressure",
        axis_units="hPa",
        time_mode="elapsed",
        time_units="hr",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x$-velocity [m/s] at ($x$ = {} km, $y$ = {} km)".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "x_label": "Elapsed time [hr]",
        "y_label": "Pressure [hPa]",
        "invert_yaxis": True,
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_height(isentropic_data):
    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_height_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # indices identifying the line to visualize
    x, y = int(grid.nx / 2), int(grid.ny / 2)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 18,
        "cbar_ticks_step": 4,
        "cbar_ticks_pos": "center",
        "cbar_center": 15,
        "cbar_orientation": "horizontal",
    }

    # instantiate the drawer
    drawer = HovmollerDiagram(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        axis_name="height",
        axis_units="km",
        time_mode="elapsed",
        time_units="hr",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x$-velocity [m/s] at ($x$ = {} km, $y$ = {} km)".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "x_label": "Elapsed time [hr]",
        "y_label": "$z$ [km]",
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
