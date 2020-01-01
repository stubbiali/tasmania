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
import matplotlib.pyplot as plt
import numpy as np
import tasmania.python.plot.plot_utils as pu

# ==================================================
# User inputs
# ==================================================
which = "expl"  # options: lazy, normal, exp

dx = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80])

figure_properties = {
    "fontsize": 16,
    "figsize": (6, 7),
    "tight_layout": True,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
    "fontsize": 16,
    "x_label": "$\\Delta x = \\Delta y$ [m]",
    "x_labelcolor": "black",
    "x_lim": (1 / 80 / 1.5, 1 / 10 * 1.5),  # (-190, 210),
    "invert_xaxis": True,
    "x_scale": "log",
    "x_ticks": (1 / 10, 1 / 20, 1 / 40, 1 / 80),
    "x_ticklabels": ("1/10", "1/20", "1/40", "1/80"),
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "$||u - u_{ex}||_2$ [m s$^{-1}$]",
    "y_labelcolor": "black",
    "y_lim": (2e-9, 2e-4),
    "invert_yaxis": False,
    "y_scale": "log",
    "y_ticks": None,
    "y_ticklabels": None,  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
    "y_tickcolor": "black",
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
    "legend_loc": "best",  #'center left',
    "legend_bbox_to_anchor": None,  # (1.04, 0.5),
    "legend_framealpha": 1.0,
    "legend_ncol": 1,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}

ax2_label = "$\\Delta t$ [s]"

ax2_scale = "log"

ax2_ticks = (1 / 10, 1 / 20, 1 / 40, 1 / 80)

ax2_ticklabels = ("1/100", "1/400", "1/1600", "1/6400")

labels = ["CC", "LCC", "PS", "SUS", "SSUS", "SSUS (CFL=2)"]

linecolors = ["red", "orange", "mediumpurple", "c", "blue", "blue"]

linestyles = ["-"] * 5 + [":"]

linewidths = [2] * 6

markers = ["s", "o", "^", "<", ">", ">"]

markersizes = [8.5] * 6

# ==================================================
# Code
# ==================================================
if __name__ == "__main__":
    if which == "lazy":
        err = [
            np.array([2.9340e-5, 4.3548e-5, 2.0848e-5, 6.4239e-6]),
            np.array([1.0545e-4, 1.1159e-4, 3.8256e-5, 1.0362e-5]),
            np.array([2.7614e-5, 5.5228e-5, 2.6481e-5, 7.9282e-6]),
            np.array([3.0470e-5, 4.8641e-5, 2.3654e-5, 7.1440e-6]),
            np.array([3.0104e-5, 4.3572e-5, 2.0852e-5, 6.2442e-6]),
        ]
    elif which == "normal":
        err = [
            np.array([8.9845e-6, 1.9629e-6, 4.7262e-8, 7.2386e-9]),
            np.array([1.0545e-4, 1.1048e-4, 3.8088e-5, 1.0351e-5]),
            np.array([2.7312e-5, 5.4049e-5, 2.6308e-5, 7.9176e-6]),
            np.array([1.0582e-5, 4.0739e-5, 2.7856e-5, 5.2777e-6]),
            np.array([9.7664e-6, 1.7651e-6, 2.2604e-8, 2.0002e-8]),
        ]
    else:
        err = [
            np.array([8.9845e-6, 1.9629e-6, 4.7262e-8, 7.2386e-9]),
            np.array([1.0545e-4, 1.1048e-4, 3.8088e-5, 1.0351e-5]),
            np.array([4.9697e-5, 1.5146e-5, 5.9153e-6, 1.7287e-6]),
            np.array([1.0892e-5, 3.6137e-6, 2.8240e-6, 9.1709e-7]),
            np.array([9.7639e-6, 1.8710e-6, 5.2504e-8, 5.1080e-9]),
            np.array([1.1997e-5, 1.8339e-6, 9.3699e-8, 1.9796e-8]),
        ]

    fig, ax = pu.get_figure_and_axes(**figure_properties)

    for i in range(len(err)):
        pu.make_lineplot(
            dx,
            err[i],
            ax,
            linecolor=linecolors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            marker=markers[i],
            markersize=markersizes[i],
            legend_label=labels[i],
        )

    pu.set_axes_properties(ax, **axes_properties)

    if False:
        ax2 = ax.twiny()
        ax2.set_xscale(ax2_scale)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax2_ticks)
        ax2.set_xticklabels(ax2_ticklabels)
        ax2.get_xaxis().set_tick_params(which="minor", size=0)
        ax2.get_xaxis().set_tick_params(which="minor", width=0)
        ax2.set_xlabel(ax2_label)

    pu.set_figure_properties(fig, **figure_properties)

    plt.show()
