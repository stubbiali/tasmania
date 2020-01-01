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
import tasmania as taz
import matplotlib.pyplot as plt

# ==================================================
# User inputs
# ==================================================
values = [
    [1377, 1020, 1235, 1275, 1230, 1620, 1360],
    [122, 68, 101, 96, 96, 153, 115],
    [111, 60, 96, 91, 93, 153, 115],
]

colors = ["darkviolet", "royalblue", "gold"]

edgecolors = ["black"] * 3

labels = ["numpy", "gtx86", "gtmc"]

bar_width = 0.2

figure_properties = {
    "fontsize": 15,
    "figsize": (7, 7),
    "tight_layout": True,
    "tight_layout_rect": None,
}

axes_properties = {
    "fontsize": 15,
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": [-0.3, 6.9],
    "invert_xaxis": False,
    "x_scale": None,
    # "x_ticks": [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25],
    "x_ticks": [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3],
    "x_ticklabels": ["FC", "LFC", "PS", "STS", "SUS", "SSUS", "SSUS-FE"],
    "x_ticklabels_rotation": 25,
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "Run time [s]",
    "y_labelcolor": "black",
    "y_lim": [0, 1800],
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": [150, 450, 750, 1050, 1350, 1650],
    "y_ticklabels": [150, 450, 750, 1050, 1350, 1650],
    "y_ticklabels_color": "black",
    "yaxis_minor_ticks_visible": False,
    "yaxis_visible": True,
    # z-axis
    "z_label": "",
    "z_labelcolor": "",
    "z_lim": None,
    "invert_zaxis": False,
    "z_scale": None,
    "z_ticks": None,
    "z_ticklabels": None,
    "z_tickcolor": "white",
    "zaxis_minor_ticks_visible": True,
    "zaxis_visible": True,
    # legend
    "legend_on": True,
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (0.5, 1),
    "legend_framealpha": 1.0,
    "legend_ncol": 3,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}

# ==================================================
# Code
# ==================================================
x = range(len(values[0]))

fig, ax = taz.get_figure_and_axes(**figure_properties)

for i in range(len(values)):
    ax.bar(
        [elx + i * bar_width for elx in x],
        values[i],
        bar_width,
        color=colors[i],
        edgecolor=edgecolors[i],
        label=labels[i],
    )

taz.set_axes_properties(ax, **axes_properties)
taz.set_figure_properties(fig, **figure_properties)

ax.xaxis.set_tick_params(length=0)
ax.xaxis.grid(False)

plt.show()
