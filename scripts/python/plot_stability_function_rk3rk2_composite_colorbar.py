# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
    "figsize": (7, 9.5),
    "tight_layout": True,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes_properties_00 = {
    "fontsize": 16,
    # title
    "title_left": "",
    "title_center": "$\\mathbf{(a)}$ RK2",
    "title_right": "",
    # x-axis
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": [-4, -3, -2, -1, 0],
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
    "y_ticks": [-3, -2, -1, 0, 1, 2, 3],
    "y_ticklabels": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
    ],  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
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
    "cmap_name": "viridis_white",
    "cbar_on": True,
    "cbar_levels": 17,
    "cbar_ticks_step": 4,
    "cbar_ticks_pos": "interface",
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_x_label": "",
    "cbar_y_label": "$\\vert r \\, (\\beta \\Delta t, \\, \\alpha \\Delta t) \\vert$",
    "cbar_title": "",
    "cbar_orientation": "vertical",
    "cbar_ax": [0, 1, 2, 3, 4, 5, 6, 7],
    "cbar_format": "%2.1f  ",
    "cbar_extend": "max",
    "cbar_extendfrac": "auto",
    "cbar_extendrect": True,
    "draw_vertical_levels": False,
}

axes_properties_01 = {
    "fontsize": 16,
    # title
    "title_left": "",
    "title_center": "$\\mathbf{(b)}$ RK3WS",
    "title_right": "",
    # x-axis
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": [-4, -3, -2, -1, 0],
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
    "y_ticks": [-3, -2, -1, 0, 1, 2, 3],
    "y_ticklabels": [],  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
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
contourf_properties_01 = {
    "fontsize": 16,
    "cmap_name": "viridis_white",
    "cbar_on": False,
    "cbar_levels": 17,
    "cbar_ticks_step": 4,
    "cbar_ticks_pos": "interface",
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_x_label": "",
    "cbar_y_label": "$\\vert r \\, (\\beta \\Delta t, \\, \\alpha \\Delta t) \\vert$",
    "cbar_title": "",
    "cbar_orientation": "vertical",
    "cbar_ax": [0, 1, 2, 3, 4, 5],
    "cbar_format": "%2.1f  ",
    "cbar_extend": "max",
    "cbar_extendfrac": "auto",
    "cbar_extendrect": True,
    "draw_vertical_levels": False,
}

axes_properties_02 = axes_properties_01.copy()
axes_properties_02[
    "title_center"
] = "$\\mathbf{(c)}$ FC ($\mathcal{E}_0 = $RK3WS)"
contourf_properties_02 = contourf_properties_01.copy()

axes_properties_03 = axes_properties_01.copy()
axes_properties_03[
    "title_center"
] = "$\\mathbf{(d)}$ LFC ($\mathcal{E}_0 = $RK3WS)"
contourf_properties_03 = contourf_properties_01.copy()

axes_properties_10 = axes_properties_00.copy()
axes_properties_10[
    "title_center"
] = "$\\mathbf{(e)}$ PS\n($\mathcal{E}_0 = $RK3WS, $\mathcal{E}_1 = $RK2)"
axes_properties_10["x_label"] = "$\\beta \\Delta t$"
axes_properties_10["x_ticklabels"] = [-4, -3, -2, -1, 0]
contourf_properties_10 = contourf_properties_01.copy()

axes_properties_11 = axes_properties_01.copy()
axes_properties_11[
    "title_center"
] = "$\\mathbf{(f)}$ STS\n($\mathcal{E}_0 = $RK3WS, $\mathcal{E}_1 = $RK2)"
axes_properties_11["x_label"] = "$\\beta \\Delta t$"
axes_properties_11["x_ticklabels"] = [-4, -3, -2, -1, 0]
contourf_properties_11 = contourf_properties_01.copy()

axes_properties_12 = axes_properties_11.copy()
axes_properties_12[
    "title_center"
] = "$\\mathbf{(g)}$ SUS\n($\mathcal{E}_0 = $RK3WS, $\mathcal{E}_1 = $RK2)"
axes_properties_12["x_label"] = "$\\beta \\Delta t$"
axes_properties_12["x_ticklabels"] = [-4, -3, -2, -1, 0]
contourf_properties_12 = contourf_properties_01.copy()

axes_properties_13 = axes_properties_11.copy()
axes_properties_13[
    "title_center"
] = "$\\mathbf{(h)}$ SSUS [$\\lambda = 1/2$]\n($\mathcal{E}_0 = $RK3WS, $\mathcal{E}_1 = \mathcal{E}^{\star}_1 = $RK2)"
axes_properties_13["x_label"] = "$\\beta \\Delta t$"
axes_properties_13["x_ticklabels"] = [-4, -3, -2, -1, 0]
contourf_properties_13 = contourf_properties_01.copy()

# ==================================================
# Code
# ==================================================
if __name__ == "__main__":
    xv = np.linspace(xlim[0], xlim[1], nx)
    yv = np.linspace(ylim[0], ylim[1], ny)
    x, y = np.meshgrid(xv, yv)

    fig, ax00 = pu.get_figure_and_axes(
        nrows=2, ncols=4, index=1, **figure_properties
    )
    _, ax01 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=2, **figure_properties
    )
    _, ax02 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=3, **figure_properties
    )
    _, ax03 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=4, **figure_properties
    )
    _, ax10 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=5, **figure_properties
    )
    _, ax11 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=6, **figure_properties
    )
    _, ax12 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=7, **figure_properties
    )
    _, ax13 = pu.get_figure_and_axes(
        fig=fig, nrows=2, ncols=4, index=8, **figure_properties
    )

    E = (
        1
        + (x + 1j * y)
        + 0.5 * (x + 1j * y) ** 2
        + 1.0 / 6.0 * (x + 1j * y) ** 3
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax01, **contourf_properties_01)
    pu.set_axes_properties(ax01, **axes_properties_01)
    pu.make_contourf(xv, yv, S, fig, ax02, **contourf_properties_02)
    pu.set_axes_properties(ax02, **axes_properties_02)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * 1j * y * (x + 1j * y)
        + 1.0 / 6.0 * (x + 1j * y) * (1j * y) ** 2
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax03, **contourf_properties_03)
    pu.set_axes_properties(ax03, **axes_properties_03)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * (x ** 2 + (1j * y) ** 2)
        + 1.0 / 6.0 * (1j * y) ** 3
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax10, **contourf_properties_10)
    pu.set_axes_properties(ax10, **axes_properties_10)

    E = (
        1
        + (x + 1j * y)
        + 0.5 * ((1j * y) ** 2 + x * 1j * y + x ** 2)
        + 1.0 / 6.0 * ((1j * y) ** 3 + 3.0 / 2.0 * x * (1j * y) ** 2)
        + 1.0 / 12.0 * (x * (1j * y) ** 3)
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax11, **contourf_properties_11)
    pu.set_axes_properties(ax11, **axes_properties_11)

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
    pu.make_contourf(xv, yv, S, fig, ax12, **contourf_properties_12)
    pu.set_axes_properties(ax12, **axes_properties_12)

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
    pu.make_contourf(xv, yv, S, fig, ax13, **contourf_properties_13)
    pu.set_axes_properties(ax13, **axes_properties_13)

    E = 1 + (x + 1j * y) + 0.5 * (x + 1j * y) ** 2
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax00, **contourf_properties_00)
    pu.set_axes_properties(ax00, **axes_properties_00)

    pu.set_figure_properties(fig, **figure_properties)

    ax00.set_visible(False)
    ax01.set_visible(False)
    ax02.set_visible(False)
    ax03.set_visible(False)
    ax10.set_visible(False)
    ax11.set_visible(False)
    ax12.set_visible(False)
    ax13.set_visible(False)

    # plt.show()
    plt.savefig("fig_colorbar.eps")
