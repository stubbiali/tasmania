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
from tasmania.python.plot.monitors import Plot, PlotComposite
from tasmania.python.plot.profile import LineProfile


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_plot_composite".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile(isentropic_data):
    # make sure the baseline directory_composer does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # fields to plot
    field1_name = "x_velocity_at_u_locations"
    field1_units = "km hr^-1"
    field2_name = "y_velocity_at_v_locations"
    field2_units = "km hr^-1"

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    #
    # Plot#1
    #
    # indices identifying the cross-line to visualize
    y, z = int(grid.ny / 2), -1

    # drawer properties
    drawer_properties = {"linecolor": "blue", "linestyle": "-", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field1_name,
        field1_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "title_left": "Left subplot",
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$x$-velocity [km/hr]",
        # 'y_lim': [0, 2.0],
        "grid_on": True,
    }

    # instantiate the left collaborator
    plot1 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # Plot#2
    #
    # indices identifying the cross-line to visualize
    y, z = int(grid.ny / 2), -1

    # drawer properties
    drawer_properties = {"linecolor": "red", "linestyle": "--", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field2_name,
        field2_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "title_left": "Right subplot",
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$y$-velocity [km/hr]",
        # 'y_lim': [0, 2.0],
        "grid_on": True,
    }

    # instantiate the right collaborator
    plot2 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # PlotComposite
    #
    # figure properties
    figure_properties = {"fontsize": 16, "figsize": (16, 7), "tight_layout": True}

    # instantiate the monitor
    monitor = PlotComposite(
        plot1,
        plot2,
        nrows=1,
        ncols=2,
        interactive=False,
        figure_properties=figure_properties,
    )

    # Plot
    monitor.store(state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_share_yaxis(isentropic_data):
    # make sure the baseline directory_composer does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_share_yaxis_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    #
    # Plot#1
    #
    # indices identifying the cross-line to visualize
    y, z = int(0.25 * grid.ny), -1

    # drawer properties
    drawer_properties = {"linecolor": "blue", "linestyle": "-", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "title_left": "$y$ = {} km".format(grid.y.to_units("km").values[y]),
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "Surface $x$-velocity [m s$^{-1}$]",
        "y_lim": [-30, 30],
        "grid_on": True,
    }

    # instantiate the left collaborator
    plot1 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # Plot#2
    #
    # indices identifying the cross-line to visualize
    y, z = int(0.75 * grid.ny), -1

    # drawer properties
    drawer_properties = {"linecolor": "red", "linestyle": "--", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field_name,
        field_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "title_left": "$y$ = {} km".format(grid.y.to_units("km").values[y]),
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_lim": [-30, 30],
        "y_ticklabels": (),
        "grid_on": True,
    }

    # instantiate the right collaborator
    plot2 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # PlotComposite
    #
    # figure properties
    figure_properties = {"fontsize": 16, "figsize": (13, 7), "tight_layout": True}

    # instantiate the monitor
    monitor = PlotComposite(
        plot1,
        plot2,
        nrows=1,
        ncols=2,
        interactive=False,
        figure_properties=figure_properties,
    )

    # plot
    monitor.store(state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_profile_share_xaxis(isentropic_data):
    # make sure the baseline directory_composer does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_profile_share_xaxis_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # fields to plot
    field1_name = "x_velocity"
    field1_units = "m s^-1"
    field2_name = "y_velocity_at_v_locations"
    field2_units = "m s^-1"

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # indices identifying the cross-line to visualize
    y, z = int(0.5 * grid.ny), -1

    #
    # Plot#1
    #
    # drawer properties
    drawer_properties = {"linecolor": "blue", "linestyle": "-", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field1_name,
        field1_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "title_center": "$y$ = {} km, $\\theta$ = {} K".format(
            grid.y.to_units("km").values[y], grid.z.to_units("K").values[z]
        ),
        # 'x_lim': [0, 500],
        "x_ticklabels": (),
        "y_label": "$x$-velocity [m/s]",
        # 'y_lim': [0, 3.0],
        "grid_on": True,
    }

    # instantiate the left collaborator
    plot1 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # Plot#2
    #
    # drawer properties
    drawer_properties = {"linecolor": "red", "linestyle": "--", "linewidth": 1.5}

    # instantiate the drawer
    drawer = LineProfile(
        grid,
        field2_name,
        field2_units,
        y=y,
        z=z,
        axis_name="x",
        axis_units="km",
        properties=drawer_properties,
    )

    # axes properties
    axes_properties = {
        "fontsize": 16,
        "x_label": "$x$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$y$-velocity [m/s]",
        # 'y_lim': [0, 2.0],
        "grid_on": True,
    }

    # instantiate the right collaborator
    plot2 = Plot(drawer, interactive=False, axes_properties=axes_properties)

    #
    # PlotComposite
    #
    # figure properties
    figure_properties = {"fontsize": 16, "figsize": (6, 9), "tight_layout": True}

    # instantiate the monitor
    monitor = PlotComposite(
        plot1,
        plot2,
        nrows=2,
        ncols=1,
        interactive=False,
        figure_properties=figure_properties,
    )

    # plot
    monitor.store(state, state, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_plot2d_r1c2(isentropic_data, drawer_topography_1d):
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_plot2d_r1c2_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # field to plot
    field1_name = "x_velocity"
    field1_units = None
    field2_name = "y_velocity"
    field2_units = None

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # index identifying the cross-section to visualize
    x = int(grid.nx / 2)

    #
    # Plot#1
    #
    # drawer properties
    drawer_properties = {
        "cmap_name": "BuRd",
        "cbar_on": False,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field1_name,
        field1_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer1 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        # 'title_left': '$x$-velocity [m s$^{-1}$]',
        "x_label": "$y$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$z$ [km]",
        "y_lim": [0, 14],
        "text": "$x$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the left collaborator
    plot1 = Plot(
        drawer, topo_drawer1, interactive=False, axes_properties=axes_properties
    )

    #
    # Plot#2
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 14,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "cbar_ax": (0, 1),
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field2_name,
        field2_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer2 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        "x_label": "$y$ [km]",
        # 'x_lim': [0, 500],
        "y_lim": [0, 14],
        "y_ticklabels": [],
        "text": "$y$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the right collaborator
    plot2 = Plot(
        drawer, topo_drawer2, interactive=False, axes_properties=axes_properties
    )

    #
    # PlotComposite
    #
    # figure properties
    figure_properties = {"fontsize": 14, "figsize": (9, 6), "tight_layout": False}

    # instantiate the monitor
    monitor = PlotComposite(
        plot1,
        plot2,
        nrows=1,
        ncols=2,
        interactive=False,
        figure_properties=figure_properties,
    )

    # plot
    monitor.store((state, state), (state, state), save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_plot2d_r2c2(isentropic_data, drawer_topography_1d):
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_plot2d_r2c2_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # field to plot
    field1_name = "x_velocity"
    field1_units = None
    field2_name = "y_velocity"
    field2_units = None

    # load data
    domain, grid_type, states = isentropic_data
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    grid.update_topography(states[-1]["time"] - states[0]["time"])
    state = states[-1]

    # index identifying the cross-section to visualize
    x = int(grid.nx / 2)

    #
    # Plot#1
    #
    # drawer properties
    drawer_properties = {
        "cmap_name": "BuRd",
        "cbar_on": False,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field1_name,
        field1_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer1 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        # 'title_left': '$x$-velocity [m s$^{-1}$]',
        "x_ticklabels": [],
        # 'x_lim': [0, 500],
        "y_label": "$z$ [km]",
        "y_lim": [0, 14],
        "text": "$x$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the left collaborator
    plot1 = Plot(
        drawer, topo_drawer1, interactive=False, axes_properties=axes_properties
    )

    #
    # Plot#2
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 14,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "cbar_ax": (0, 1),
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field2_name,
        field2_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer2 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        "x_ticklabels": [],
        "y_lim": [0, 14],
        "y_ticklabels": [],
        "text": "$y$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the right collaborator
    plot2 = Plot(
        drawer, topo_drawer2, interactive=False, axes_properties=axes_properties
    )

    #
    # Plot#3
    #
    # drawer properties
    drawer_properties = {
        "cmap_name": "BuRd",
        "cbar_on": False,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field1_name,
        field1_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer3 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        # 'title_left': '$x$-velocity [m s$^{-1}$]',
        "x_label": "$y$ [km]",
        # 'x_lim': [0, 500],
        "y_label": "$z$ [km]",
        "y_lim": [0, 14],
        "text": "$x$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the left collaborator
    plot3 = Plot(
        drawer, topo_drawer3, interactive=False, axes_properties=axes_properties
    )

    #
    # Plot#4
    #
    # drawer properties
    drawer_properties = {
        "fontsize": 14,
        "cmap_name": "BuRd",
        "cbar_on": True,
        "cbar_levels": 22,
        "cbar_ticks_step": 2,
        "cbar_center": 15,
        "cbar_half_width": 10.5,
        "cbar_ax": (2, 3),
        "linecolor": "black",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = Contourf(
        grid,
        field2_name,
        field2_units,
        x=x,
        yaxis_units="km",
        zaxis_name="height",
        zaxis_units="km",
        properties=drawer_properties,
    )

    # instantiate the drawer plotting the topography
    topo_drawer4 = drawer_topography_1d(
        grid, topography_units="km", x=x, axis_units="km"
    )

    # axes properties
    axes_properties = {
        "fontsize": 14,
        "x_label": "$y$ [km]",
        # 'x_lim': [0, 500],
        "y_lim": [0, 14],
        "y_ticklabels": [],
        "text": "$y$-velocity",
        "text_loc": "upper right",
    }

    # instantiate the right collaborator
    plot4 = Plot(
        drawer, topo_drawer4, interactive=False, axes_properties=axes_properties
    )

    #
    # PlotComposite
    #
    # figure properties
    figure_properties = {"fontsize": 14, "figsize": (9, 12), "tight_layout": False}

    # instantiate the monitor
    monitor = PlotComposite(
        plot1,
        plot2,
        plot3,
        plot4,
        nrows=2,
        ncols=2,
        interactive=False,
        figure_properties=figure_properties,
    )

    # plot
    monitor.store(*((state, state),) * 4, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
