# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from tasmania.python.plot.profile import LineProfile


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_profile".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_x(isentropic_data):
    # field to plot
    field_name = "x_velocity"
    field_units = "km hr^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_x_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # indices identifying the cross-line to visualize
    y, z = int(grid.ny / 2), -1

    # drawer properties
    drawer_properties = {
        "linecolor": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        y=y,
        z=z,
        axis_units="km",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$y = ${} km, $\\theta = ${} K".format(
            grid.y.to_units("km").values[y], grid.z.to_units("K").values[z]
        ),
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$x$-velocity [km hr$^{-1}$]",
        # 'y_lim': [0, 2.0],
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
    monitor.store(state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_y(isentropic_data):
    # field to plot
    field_name = "y_velocity_at_v_locations"
    field_units = "m s^-1"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_y_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # indices identifying the cross-line to visualize
    x, z = int(grid.nx / 2), -1

    # drawer properties
    drawer_properties = {
        "linecolor": "red",
        "linestyle": "--",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        x=x,
        z=z,
        axis_units="km",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x = ${} km, $\\theta = ${} K".format(
            grid.x.to_units("km").values[x], grid.z.to_units("K").values[z]
        ),
        "x_label": "$y$ [km]",
        "x_lim": None,
        "y_label": "$y$-velocity [m s$^{-1}$]",
        "y_lim": [-4.5, 4.5],
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
    monitor.store(state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_z(isentropic_data):
    # field to plot
    field_name = "air_pressure_on_interface_levels"
    field_units = "atm"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_z_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # indices identifying the cross-line to visualize
    x, y = int(grid.nx / 3), int(0.75 * grid.ny)

    # drawer properties
    drawer_properties = {
        "linecolor": "lightblue",
        "linestyle": "-",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        axis_units="K",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x = ${} km, $y = ${} km".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "x_label": "Air pressure [atm]",
        # 'x_lim': [0, 0.2],
        "y_label": "$\\theta$ [K]",
        "y_lim": None,
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
    monitor.store(state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_h(isentropic_data):
    # field to plot
    field_name = "air_pressure_on_interface_levels"
    field_units = "kPa"

    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_h_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # indices identifying the cross-line to visualize
    x, y = int(grid.nx / 3), int(0.75 * grid.ny)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "linecolor": "lightblue",
        "linestyle": "-",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        x=x,
        y=y,
        axis_name="height",
        axis_units="km",
        properties=drawer_properties,
    )

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "title_left": "$x = ${} km, $y = ${} km".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "x_label": "Air pressure [kPa]",
        # 'x_lim': [0, 0.2],
        "y_label": "$z$ [km]",
        "y_lim": None,
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
    monitor.store(state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
