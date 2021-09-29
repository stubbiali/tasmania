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
from tasmania.python.plot.spectrals import CDF


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_spectrals".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_cdf_u(isentropic_data):
    # field to plot
    field_name = "x_velocity_at_u_locations"
    field_units = "m s^-1"

    # make sure the folder tests/baseline_images/test_timeseries does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_cdf_u_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # drawer properties
    drawer_properties = {
        "number_of_bins": 500,
        "data_on_xaxis": True,
        "linecolor": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
    }

    # instantiate the drawer
    drawer = CDF(grid, field_name, field_units, properties=drawer_properties)

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "x_label": "$x$-velocity [m s$^{-1}$]",
        "y_label": "CDF",
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_cdf_qc(isentropic_data):
    # field to plot
    field_name = "mass_fraction_of_cloud_liquid_water_in_air"
    field_units = "g kg^-1"

    # make sure the folder tests/baseline_images/test_timeseries does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_cdf_qc_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # grab data from dataset
    domain, grid_type, states = isentropic_data
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    # drawer properties
    drawer_properties = {
        "number_of_bins": 1000,
        "data_on_xaxis": False,
        "linecolor": "red",
        "linestyle": ":",
        "linewidth": 2.5,
    }

    # instantiate the drawer
    drawer = CDF(grid, field_name, field_units, properties=drawer_properties)

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 8),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "x_label": "CDF",
        "y_label": "$q_c$ [g kg$^{-1}$]",
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
        drawer(state)
    monitor.store(states[-1], save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
