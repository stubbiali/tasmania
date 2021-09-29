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
from tasmania.python.plot.patches import Circle, Rectangle


baseline_dir = os.path.join(
    os.getcwd(),
    "baseline_images/py{}{}/test_patches".format(
        sys.version_info.major, sys.version_info.minor
    ),
)


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_circle():
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_circle_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "xy": (0.5, 0.5),
        "radius": 0.25,
        "linewidth": 3.5,
        "edgecolor": "green",
        "facecolor": "yellow",
    }

    # instantiate the drawer
    drawer = Circle(properties=drawer_properties)

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "x_label": "x",
        "x_lim": (0, 1),
        "y_label": "y",
        "y_lim": (0, 1),
    }

    # instantiate the monitor
    monitor = Plot(
        drawer,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    monitor.store({}, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_rectangle():
    # make sure the baseline directory does exist
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    # make sure the baseline image will exist at the end of this run
    save_dest = os.path.join(baseline_dir, "test_rectangle_nompl.eps")
    if os.path.exists(save_dest):
        os.remove(save_dest)

    # drawer properties
    drawer_properties = {
        "fontsize": 16,
        "xy": (0.2, 0.45),
        "width": 0.5,
        "height": 0.1,
        "angle": 45,
        "linewidth": 2.5,
        "edgecolor": "black",
        "facecolor": "blue",
    }

    # instantiate the drawer
    drawer = Rectangle(properties=drawer_properties)

    # figure and axes properties
    figure_properties = {
        "fontsize": 16,
        "figsize": (7, 7),
        "tight_layout": True,
    }
    axes_properties = {
        "fontsize": 16,
        "x_label": "x",
        "x_lim": (0, 1),
        "y_label": "y",
        "y_lim": (0, 1),
    }

    # instantiate the monitor
    monitor = Plot(
        drawer,
        interactive=False,
        figure_properties=figure_properties,
        axes_properties=axes_properties,
    )

    # plot
    monitor.store({}, save_dest=save_dest)

    assert os.path.exists(save_dest)

    return monitor.figure


if __name__ == "__main__":
    pytest.main([__file__])
