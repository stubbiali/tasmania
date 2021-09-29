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
import numpy as np
import tasmania as taz
import matplotlib.pyplot as plt

# ==================================================
# User inputs
# ==================================================
values = np.array(
    [
        [664.38, 541.38, 616.82, 752.5],
        [843.86, 713.04, 793.45, 931.89],
        [24.73, 19.45, 25.36, 35.33],
        [19.25, 16.91, 21.20, 28.36],
        [6.19, 5.33, 7.97, 11.3],
    ]
)
values = values[2:, :]
group_dim = 0  # values belonging to the same group of bars
color_dim = 1  # values represented with the same color

colors = ["orange", "red", "blue", "cyan", "green"]
colors = colors[2:]

edgecolors = ["black"] * 5

labels = ["numpy", "gt-numpy", "gt-x86", "gt-mc", "gt-cuda"]
labels = labels[2:]

bar_width = 0.2

figure_properties = {
    "fontsize": 15,
    "figsize": (7, 6),
    "tight_layout": True,
    "tight_layout_rect": None,
}

axes_properties = {
    "fontsize": 15,
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": [-0.3, 4.7],
    "invert_xaxis": False,
    "x_scale": None,
    # "x_ticks": [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25],
    "x_ticks": [0.4, 1.6, 2.8, 4.0],
    "x_ticklabels": ["FC", "LFC", "SUS", "SSUS"],
    "x_ticklabels_rotation": 25,
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "Run time [s]",
    "y_labelcolor": "black",
    "y_lim": [0, 40],
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": None,
    "y_ticklabels": None,
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
bars_per_group = values.shape[group_dim]
bars_per_color = values.shape[color_dim]

ticks_distance = bar_width * (bars_per_group + 1)
ticks = np.linspace(0, (bars_per_color - 1) * ticks_distance, bars_per_color)

fig, ax = taz.get_figure_and_axes(**figure_properties)

for i in range(bars_per_group):
    ax.bar(
        [elx + i * bar_width for elx in ticks],
        values[i, :] if color_dim == 1 else values[:, i],
        bar_width,
        color=colors[i],
        edgecolor=edgecolors[i],
        label=labels[i],
    )

axes_properties["x_lim"] = (
    ticks[0] - 1.5 * bar_width,
    ticks[-1] + (bars_per_group + 0.5) * bar_width,
)
axes_properties["x_ticks"] = tuple(
    tick + (bars_per_group / 2.0 - 0.5) * bar_width for tick in ticks
)
taz.set_axes_properties(ax, **axes_properties)
taz.set_figure_properties(fig, **figure_properties)

ax.xaxis.set_tick_params(length=0)
ax.xaxis.grid(False)

plt.show()
