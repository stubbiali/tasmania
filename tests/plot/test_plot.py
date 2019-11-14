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
import os
import pytest
import sys

from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.monitors import Plot
from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.quiver import Quiver


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_plot".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_x(isentropic_data):
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_x_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # fields to plot
    field1_name = "x_velocity_at_u_locations"
    field1_units = "km hr^-1"
    field2_name = "y_velocity_at_v_locations"
    field2_units = "km hr^-1"

    # indices identifying the cross-line to visualize
    y, z = int(grid.ny / 2), -1

    #
    # Drawer#1
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "linecolor": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
        "legend_label": "$x$-velocity",
    }

    # instantiate the drawer
    drawer1 = LineProfile(
        grid,
        field1_name,
        field1_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    #
    # Drawer#2
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "linecolor": "green",
        "linestyle": "--",
        "linewidth": 1.5,
        "legend_label": "$y$-velocity",
    }

    # instantiate the drawer
    drawer2 = LineProfile(
        grid,
        field2_name,
        field2_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    #
    # Plot
    #
    # figure and axes properties
    figure_properties = {"fontsize": 16, "figsize": (7, 7), "tight_layout": True}
    axes_properties = {
        "fontsize": 16,
        "title_left": "$y = ${} km, $\\theta = ${} K".format(
            grid.y.to_units("km").values[y], grid.z.to_units("K").values[z]
        ),
        "title_right": str(state["time"] - states[0]["time"]),
        "x_label": "$x$ [km]",
        #'x_lim': [0, 500],
        "y_label": "Velocity [m/s]",
        "y_lim": [-20, 100],
        "legend_on": True,
        "legend_loc": "best",
        "legend_framealpha": 1.0,
        "grid_on": True,
    }

    # instantiate the monitor
    monitor = Plot(
        drawer1,
        drawer2,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    monitor.store(state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_z(isentropic_data):
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_z_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # fields to plot
    field1_name = "x_velocity_at_u_locations"
    field1_units = "km hr^-1"
    field2_name = "y_velocity_at_v_locations"
    field2_units = "km hr^-1"

    # indices identifying the cross-line to visualize
    x, y = int(grid.nx / 2), int(grid.ny / 2)

    #
    # Drawer#1
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "linecolor": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
        "legend_label": "$x$-velocity",
    }

    # instantiate the monitor
    drawer1 = LineProfile(
        grid,
        field1_name,
        field1_units,
        x=x,
        y=y,
        axis_name="z",
        axis_units="K",
        properties=drawer_properties,
    )

    #
    # Drawer#2
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "linecolor": "lightblue",
        "linestyle": "--",
        "linewidth": 1.5,
        "legend_label": "$y$-velocity",
    }

    # instantiate the monitor
    drawer2 = LineProfile(
        grid,
        field2_name,
        field2_units,
        x=x,
        y=y,
        axis_name="z",
        axis_units="K",
        properties=drawer_properties,
    )

    #
    # Plot
    #
    # figure and axes properties
    figure_properties = {"fontsize": 16, "figsize": (7, 7), "tight_layout": True}
    axes_properties = {
        "fontsize": 16,
        "title_center": "$x$ = {} km, $y$ = {} km".format(
            grid.x.to_units("km").values[x], grid.y.to_units("km").values[y]
        ),
        "title_right": str(state["time"] - states[0]["time"]),
        "x_label": "Velocity [m/s]",
        "x_lim": [-20, 100],
        "y_label": "$\\theta$ [K]",
        "legend_on": True,
        "legend_loc": "best",
        "grid_on": True,
    }

    # instantiate the monitor
    monitor = Plot(
        drawer1,
        drawer2,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    monitor.store(state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_plot_2d(isentropic_data, drawer_topography_2d):
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_plot_2d_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # index identifying the cross-section
    z = -1

    #
    # Drawer#1
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "cmap_name": "BuRd",
        "cbar_levels": 14,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 7.5,
        "cbar_x_label": "",
        "cbar_y_label": "",
        "cbar_title": "",
        "cbar_orientation": "horizontal",
    }

    # instantiate the drawer
    drawer1 = Contourf(
        grid,
        "horizontal_velocity",
        "m s^-1",
        z=z,
        xaxis_units="km",
        yaxis_units="km",
        properties=drawer_properties,
    )

    #
    # Drawer#2
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "x_step": 2,
        "y_step": 2,
        "cmap_name": None,
        "alpha": 0.5,
    }

    # instantiate the monitor
    drawer2 = Quiver(
        grid,
        z=z,
        xcomp_name="x_velocity",
        ycomp_name="y_velocity",
        xaxis_units="km",
        yaxis_units="km",
        properties=drawer_properties,
    )

    #
    # Drawer#3
    #
    topo_drawer = drawer_topography_2d(grid, xaxis_units="km", yaxis_units="km")

    #
    # Plot
    #
    # figure and axes properties
    figure_properties = {"fontsize": 16, "figsize": (7, 8), "tight_layout": True}
    axes_properties = {
        "fontsize": 16,
        "title_left": "Horizontal velocity [m s$^{-1}$] at the surface",
        "title_right": str(state["time"] - states[0]["time"]),
        "x_label": "$x$ [km]",
        #'x_lim': [0, 500],
        "y_label": "$y$ [km]",
        #'y_lim': [-250, 250],
    }

    # instantiate the monitor
    monitor = Plot(
        drawer1,
        drawer2,
        topo_drawer,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    monitor.store(state, state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
