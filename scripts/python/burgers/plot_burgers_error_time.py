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
which = "normal"  # options: lazy, midlazy, normal

dx = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80])

figure_properties = {
    "fontsize": 16,
    "figsize": (13.5, 6.5),
    "tight_layout": False,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes1_properties = {
    "fontsize": 16,
    "x_label": "$\\Delta x = \\Delta y$ [m]",
    "x_labelcolor": "black",
    "x_lim": (1 / 80 / 1.7, 1 / 10 * 1.7),  # (-190, 210),
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
    "y_lim": (5.5e-10, 2e-3),
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
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (1.06, 1.175),  # (0.635, 1.175),
    "legend_framealpha": 1.0,
    "legend_ncol": 3,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}

axes2_properties = {
    "fontsize": 16,
    "x_label": "$\\Delta x = \\Delta y$ [m]",
    "x_labelcolor": "black",
    "x_lim": (1 / 80 / 1.7, 1 / 10 * 1.7),  # (-190, 210),
    "invert_xaxis": True,
    "x_scale": "log",
    "x_ticks": (1 / 10, 1 / 20, 1 / 40, 1 / 80),
    "x_ticklabels": ("1/10", "1/20", "1/40", "1/80"),
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "Compute time [s]",
    "y_labelcolor": "black",
    "y_lim": (1 / 1.5, 500 * 1.5),
    "invert_yaxis": False,
    "y_scale": "log",
    "y_ticks": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    "y_ticklabels": (
        "$2^0$",
        "$2^1$",
        "$2^2$",
        "$2^3$",
        "$2^4$",
        "$2^5$",
        "$2^6$",
        "$2^7$",
        "$2^8$",
        "$2^9$",
    ),
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
    "legend_on": False,
    "legend_loc": "best",  # 'center left',
    "legend_bbox_to_anchor": None,  # (1.04, 0.5),
    "legend_framealpha": 1.0,
    "legend_ncol": 1,
    # textbox
    "text": None,  # '$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}

labels = [
    "ConcurrentCoupling",
    "LazyConcurrentCoupling",
    "ParallelSplitting",
    "SequentialUpdateSplitting",
    #'SymmetrizedSequentialSplitting                                                                                ',
    "SymmSequentialUpdateSplitting",
    "SymmSequentialUpdateSplitting (CFL=2)",
]

linecolors = ["red", "orange", "mediumpurple", "green", "blue", "blue"]
# linecolors = ['green', 'blue', 'blue']

linestyles = ["-"] * 5 + [":"]
# linestyles = ['-', '-', ':']

linewidths = [2] * 6

markers = ["s", "o", "^", "<", ">", ">"]
# markers = ['<', '>', '>']

markersizes = [8.5] * 6

markeredgecolors = linecolors

markerfacecolors = linecolors

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
        time = [
            np.array([1, 5, 25, 120], dtype=np.float32),
            np.array([1, 5, 21, 97], dtype=np.float32),
            np.array([3, 15, 61, 270], dtype=np.float32),
            np.array([3, 12, 50, 219], dtype=np.float32),
            np.array([5, 20, 83, 351], dtype=np.float32),
        ]
    elif which == "midlazy":
        err = [
            np.array([8.9845e-6, 1.9629e-6, 4.7262e-8, 7.2386e-9]),
            np.array([1.0545e-4, 1.1048e-4, 3.8088e-5, 1.0351e-5]),
            np.array([2.7312e-5, 5.4049e-5, 2.6308e-5, 7.9176e-6]),
            np.array([1.0582e-5, 4.0739e-5, 2.7856e-5, 5.2777e-6]),
            np.array([9.7664e-6, 1.7651e-6, 2.2604e-8, 2.0002e-8]),
        ]
        time = [
            np.array([2, 8, 34, 155], dtype=np.float32),
            np.array([1, 7, 30, 135], dtype=np.float32),
            np.array([4, 16, 69, 302], dtype=np.float32),
            np.array([3, 14, 59, 295], dtype=np.float32),
            np.array([6, 23, 99, 400], dtype=np.float32),
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
        time = [
            np.array([2, 7, 31.5, 147], dtype=np.float32),
            np.array([1, 7, 28, 130], dtype=np.float32),
            np.array([4, 16, 67, 290], dtype=np.float32),
            np.array([3, 14, 58, 252], dtype=np.float32),
            np.array([5, 22, 93, 400], dtype=np.float32),
            np.array([2, 11, 46, 199], dtype=np.float32),
        ]

    fig, ax1 = pu.get_figure_and_axes(ncols=2, nrows=1, index=1, **figure_properties)

    for i in range(len(err)):
        pu.make_lineplot(
            dx,
            err[i],
            ax1,
            linecolor=linecolors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            marker=markers[i],
            markersize=markersizes[i],
            markeredgecolor=markeredgecolors[i],
            markerfacecolor=markerfacecolors[i],
            legend_label=labels[i],
        )

    pu.make_lineplot(
        dx, 1e-1 * dx ** 2, ax1, linecolor="black", linestyle="--", linewidth=1.5
    )
    pu.make_lineplot(
        dx,
        1e-9 * (dx / dx[-1]) ** 4,
        ax1,
        linecolor="black",
        linestyle="--",
        linewidth=1.5,
    )

    plt.text(
        dx[-1],
        1e-1 * dx[-1] ** 2,
        " $\\mathcal{O}(\\Delta t)$",
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.text(
        dx[-1],
        1e-9,
        " $\\mathcal{O}(\\Delta t^2)$",
        horizontalalignment="left",
        verticalalignment="center",
    )

    _, ax2 = pu.get_figure_and_axes(fig, ncols=2, nrows=1, index=2, **figure_properties)

    for i in range(len(time)):
        pu.make_lineplot(
            dx,
            time[i],
            ax2,
            linecolor=linecolors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            marker=markers[i],
            markersize=markersizes[i],
            markeredgecolor=markeredgecolors[i],
            markerfacecolor=markerfacecolors[i],
            legend_label=labels[i],
        )

    pu.set_axes_properties(ax1, **axes1_properties)
    pu.set_axes_properties(ax2, **axes2_properties)

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
