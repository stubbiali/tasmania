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
xlim = [-9, 0]  # x = beta*dt
nx = 501
ylim = [-9, 9]  # y = alpha*dt
ny = 1001

figure_properties = {
    "fontsize": 16,
    "figsize": (14, 4.75),
    "tight_layout": True,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
    "subplots_adjust_right": 1.05,
}

axes_properties_00 = {
    "fontsize": 16,
    # title
    "title_left": "",
    "title_center": "$\\mathbf{(a)}$ LFC ($\\mathcal{E}_0 = $IE)",
    "title_right": "",
    # x-axis
    "x_label": "$\\beta \\Delta t$",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": [-9, -6, -3, 0],
    "x_ticklabels": [-9, -6, -3, 0],
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "$\\alpha \\Delta t$",
    "y_labelcolor": "black",
    "y_lim": ylim,
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": [-9, -6, -3, 0, 3, 6, 9],
    "y_ticklabels": [
        -9,
        -6,
        -3,
        0,
        3,
        6,
        9,
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
    "cbar_y_label": "$\\vert r \\, (\\beta \\Delta t, \\, \\alpha \\Delta t) \\vert$",
    "cbar_x_label": "",
    "cbar_title": "",
    "cbar_orientation": "vertical",
    "cbar_ax": [0, 1, 2, 3],
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
    "title_center": "$\\mathbf{(b)}$ PS ($\\mathcal{E}_0 = \\mathcal{E}_1 = $IE)",
    "title_right": "",
    # x-axis
    "x_label": "$\\beta \\Delta t$",
    "x_labelcolor": "black",
    "x_lim": xlim,  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": [-9, -6, -3, 0],
    "x_ticklabels": [-9, -6, -3, 0],
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "",
    "y_labelcolor": "black",
    "y_lim": ylim,
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": [-9, -6, -3, 0, 3, 6, 9],
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
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_extend": "max",
    "cbar_extendfrac": "auto",
    "cbar_extendrect": True,
    "draw_vertical_levels": False,
}

axes_properties_02 = axes_properties_01.copy()
axes_properties_02[
    "title_center"
] = "$\\mathbf{(c)}$ PS ($\\mathcal{E}_0 = \\mathcal{E}_1 = $TR)"
contourf_properties_02 = contourf_properties_01.copy()

axes_properties_03 = axes_properties_01.copy()
axes_properties_03[
    "title_center"
] = "$\\mathbf{(d)}$ STS ($\\mathcal{E}_0 = \\mathcal{E}_1 = $IE)"
contourf_properties_03 = contourf_properties_01.copy()

# ==================================================
# Code
# ==================================================
if __name__ == "__main__":
    xv = np.linspace(xlim[0], xlim[1], nx)
    yv = np.linspace(ylim[0], ylim[1], ny)
    x, y = np.meshgrid(xv, yv)

    fig, ax00 = pu.get_figure_and_axes(
        nrows=1, ncols=4, index=1, **figure_properties
    )
    _, ax01 = pu.get_figure_and_axes(
        fig=fig, nrows=1, ncols=4, index=2, **figure_properties
    )
    _, ax02 = pu.get_figure_and_axes(
        fig=fig, nrows=1, ncols=4, index=3, **figure_properties
    )
    _, ax03 = pu.get_figure_and_axes(
        fig=fig, nrows=1, ncols=4, index=4, **figure_properties
    )
    # fig, [ax00, ax01, ax02] = plt.subplots(
    #     1, 3, figsize=figure_properties["figsize"]
    # )

    E = 1 / (1 - x) + 1 / (1 - 1j * y) - 1
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax01, **contourf_properties_01)
    pu.set_axes_properties(ax01, **axes_properties_01)

    E = (
        (1 + 0.5 * x) / (1 - 0.5 * x)
        + (1 + 0.5 * 1j * y) / (1 - 0.5 * 1j * y)
        - 1
    )
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax02, **contourf_properties_02)
    pu.set_axes_properties(ax02, **axes_properties_02)

    E = 1 / ((1 - x) * (1 - 1j * y))
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax03, **contourf_properties_03)
    pu.set_axes_properties(ax03, **axes_properties_03)

    pu.set_figure_properties(fig, **figure_properties)

    E = (1 + x) / (1 - 1j * y)
    S = np.abs(E)
    pu.make_contourf(xv, yv, S, fig, ax00, **contourf_properties_00)
    pu.set_axes_properties(ax00, **axes_properties_00)

    # ax00.set_visible(False)
    # ax01.set_visible(False)
    # ax02.set_visible(False)
    # ax03.set_visible(False)

    plt.show()
    # fig.savefig(
    #     "/Users/subbiali/Desktop/phd/manuscripts/pdc-paper/revision2/img/theory/test2.eps"
    # )
