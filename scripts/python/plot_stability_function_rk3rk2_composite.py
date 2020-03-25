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
xlim = [-4, 0]  # x = beta*dt
nx = 101
ylim = [-3, 3]  # y = alpha*dt
ny = 201

figure_properties = {
    "fontsize": 16,
    "figsize": (14, 7),
    "tight_layout": False,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes_properties_00 = {
    "fontsize": 16,
    # title
    "title_left": "",
    "title_center": "$\\mathbf{(a)}$ FC",
    "title_right": "",
    # x-axis
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": range(int(xlim[0]), int(xlim[1])+1),
    "x_ticklabels": [],
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "$\\alpha \\Delta t$",
    "y_labelcolor": "black",
    "y_lim": ylim,
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": range(int(ylim[0]), int(ylim[1])+1),
    "y_ticklabels": None,  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
    "y_tickcolor": "black",
    "yaxis_minor_ticks_visible": False,
    "yaxis_visible": True,
    # legend
    "legend_on": False,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}
contourf_properties_00 = {
    "fontsize": 16,
    "cmap_name": "jet",
    "cbar_on": True,
    "cbar_levels": 21,
    "cbar_ticks_step": 2,
    "cbar_ticks_pos": "interface",
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_x_label": "",
    "cbar_y_label": "$\\mathcal{S} \\, (\\beta \\Delta t, \\, \\alpha \\Delta t)$",
    "cbar_title": "",
    "cbar_orientation": "vertical",
    "cbar_ax": [0, 1, 2, 3, 4, 5],
    "cbar_format": "%2.1f  ",
    "cbar_extend": False,
    "draw_vertical_levels": False,
}

axes_properties_01 = {
    "fontsize": 16,
    # title
    "title_left": "",
    "title_center": "$\\mathbf{(b)}$ LFC",
    "title_right": "",
    # x-axis
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": range(int(xlim[0]), int(xlim[1])+1),
    "x_ticklabels": [],
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "",
    "y_labelcolor": "black",
    "y_lim": ylim,
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": range(int(ylim[0]), int(ylim[1])+1),
    "y_ticklabels": [],  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
    "y_tickcolor": "black",
    "yaxis_minor_ticks_visible": False,
    "yaxis_visible": True,
    # legend
    "legend_on": True,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    # grid
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}
contourf_properties_01 = {
    "fontsize": 16,
    "cmap_name": "jet",
    "cbar_on": False,
    "cbar_levels": 21,
    "cbar_ticks_step": 4,
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_extend": False,
    "draw_vertical_levels": False,
}

axes_properties_02 = axes_properties_01.copy()
axes_properties_02["title_center"] = "$\\mathbf{(c)}$ PS"
contourf_properties_02 = contourf_properties_01.copy()

axes_properties_10 = axes_properties_00.copy()
axes_properties_10["title_center"] = "$\\mathbf{(d)}$ STS"
axes_properties_10["x_label"] = "$\\beta \\Delta t$"
axes_properties_10["x_ticklabels"] = None
contourf_properties_10 = contourf_properties_01.copy()

axes_properties_11 = axes_properties_01.copy()
axes_properties_11["title_center"] = "$\\mathbf{(e)}$ SUS"
axes_properties_11["x_label"] = "$\\beta \\Delta t$"
axes_properties_11["x_ticklabels"] = None
contourf_properties_11 = contourf_properties_01.copy()

axes_properties_12 = axes_properties_11.copy()
axes_properties_12["title_center"] = "$\\mathbf{(f)}$ SSUS"
contourf_properties_12 = contourf_properties_01.copy()

# ==================================================
# Code
# ==================================================
if __name__ == "__main__":
    xv = np.linspace(xlim[0], xlim[1], nx)
    yv = np.linspace(ylim[0], ylim[1], ny)
    x, y = np.meshgrid(xv, yv)

    fig, ax00 = pu.get_figure_and_axes(nrows=2, ncols=3, index=1, **figure_properties)
    _, ax01 = pu.get_figure_and_axes(fig=fig, nrows=2, ncols=3, index=2, **figure_properties)
    _, ax02 = pu.get_figure_and_axes(fig=fig, nrows=2, ncols=3, index=3, **figure_properties)
    _, ax10 = pu.get_figure_and_axes(fig=fig, nrows=2, ncols=3, index=4, **figure_properties)
    _, ax11 = pu.get_figure_and_axes(fig=fig, nrows=2, ncols=3, index=5, **figure_properties)
    _, ax12 = pu.get_figure_and_axes(fig=fig, nrows=2, ncols=3, index=6, **figure_properties)

    E = 1 + (x + 1j * y) + 0.5 * (x + 1j * y) ** 2 + 1.0 / 6.0 * (x + 1j * y) ** 3
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax00, **contourf_properties_00)
    pu.set_axes_properties(ax00, **axes_properties_00)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * 1j * y * (x + 1j * y)
        + 1.0 / 6.0 * (x + 1j * y) * (1j * y) ** 2
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax01, **contourf_properties_01)
    pu.set_axes_properties(ax01, **axes_properties_01)

    E = 1 + (x + 1j * y) + 0.5 * (x ** 2 + (1j * y) ** 2) + 1.0 / 6.0 * (1j * y) ** 3
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax02, **contourf_properties_02)
    pu.set_axes_properties(ax02, **axes_properties_02)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * ((1j * y) ** 2 + x * 1j * y + x ** 2)
        + 1.0 / 6.0 * ((1j * y) ** 3 + 3.0 / 2.0 * x * (1j * y) ** 2)
        + 1.0 / 12.0 * (x * (1j * y) ** 3)
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax10, **contourf_properties_10)
    pu.set_axes_properties(ax10, **axes_properties_10)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * (x + 1j * y) ** 2
        + 1.0 / 6.0 * ((x + 1j * y) ** 3 - x ** 3)
        + 1.0 / 6.0 * x * (1j * y) ** 3
        + 0.25 * (x ** 2) * ((1j * y) ** 2)
        + 1.0 / 12.0 * (x ** 2) * ((1j * y) ** 3)
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax11, **contourf_properties_11)
    pu.set_axes_properties(ax11, **axes_properties_11)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * (x + 1j * y) ** 2
        + 1.0 / 6.0 * ((x + 1j * y) ** 3 - 0.25 * x ** 3)
        + 1.0
        / 24.0
        * (
            4.0 * x * (1j * y) ** 3
            + 6.0 * (x ** 2) * (1j * y) ** 2
            + 3.0 * 1j * y * (x ** 3)
            + 3.0 / 8.0 * x ** 4
        )
        + 1.0
        / 48.0
        * (
            4.0 * (x ** 2) * (1j * y) ** 3
            + 3.0 * (x ** 3) * (1j * y) ** 2
            + 0.75 * (x ** 4) * 1j * y
        )
        + 1.0
        / 96.0
        * (2.0 * (x ** 3) * (1j * y) ** 3 + 0.75 * (x ** 4) * (1j * y) ** 2)
        + 1.0 / 384.0 * (x ** 4) * (1j * y) ** 3
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax12, **contourf_properties_12)
    pu.set_axes_properties(ax12, **axes_properties_12)

    pu.set_figure_properties(fig, **figure_properties)
    plt.show()
