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
coupling_method = "sus"  # options: cc, lcc, ps, sus

xlim = [0, 3]  # x = beta*dt
nx = 101
ylim = [-6, 6]  # y = alpha*dt
ny = 201

figure_properties = {
    "fontsize": 16,
    "figsize": (6, 7),
    "tight_layout": True,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
    "fontsize": 16,
    "title_center": "",
    "title_left": "",
    "title_right": "",
    "x_label": "$\\beta \\Delta t$",  #'Time (UTC)',
    "x_lim": (0, 3),  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": None,
    "x_ticklabels": None,
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    "y_label": "$\\alpha \\Delta t$",
    "y_lim": (-3, 3),
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": None,
    "y_ticklabels": None,  # ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
    "yaxis_minor_ticks_visible": False,
    "yaxis_visible": True,
    "z_label": "",
    "z_lim": None,
    "invert_zaxis": False,
    "z_scale": None,
    "z_ticks": None,
    "z_ticklabels": None,
    "zaxis_minor_ticks_visible": True,
    "zaxis_visible": True,
    "legend_on": False,
    "legend_loc": "best",  #'center left',
    "legend_bbox_to_anchor": None,  # (1.04, 0.5),
    "legend_framealpha": 1.0,
    "legend_ncol": 1,
    "text": "SUS",  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper right",
    "grid_on": True,
    "grid_properties": {"linestyle": ":"},
}

contourf_properties = {
    "fontsize": 16,
    "cmap_name": "jet",
    "cbar_on": True,
    "cbar_levels": 17,
    "cbar_ticks_step": 4,
    "cbar_ticks_pos": "interface",
    "cbar_center": 0.5,
    "cbar_half_width": 0.5,
    "cbar_x_label": "$| E(\\beta \\Delta t, \\, \\alpha \\Delta t) |$",
    "cbar_y_label": "",
    "cbar_title": "",
    "cbar_orientation": "horizontal",
    "cbar_ax": None,
    "draw_vertical_levels": False,
}

# ==================================================
# Code
# ==================================================
if __name__ == "__main__":
    xv = np.linspace(xlim[0], xlim[1], nx)
    yv = np.linspace(ylim[0], ylim[1], ny)
    x, y = np.meshgrid(xv, yv)

    if coupling_method == "cc":
        E = 1 - (x + 1j * y) + 0.5 * (x + 1j * y) ** 2
    elif coupling_method == "lcc":
        E = (
            1
            - (x + 1j * y)
            + 0.5 * 1j * y * (x + 1j * y)
            - 1.0 / 6.0 * (x + 1j * y) * (1j * y) ** 2
        )
    elif coupling_method == "ps":
        E = (
            1
            - (x + 1j * y)
            + 0.5 * (x ** 2 + (1j * y) ** 2)
            - 1.0 / 6.0 * (1j * y) ** 3
        )
    elif coupling_method == "sus":
        E = (
            1
            - (x + 1j * y)
            + 0.5 * (x + 1j * y) ** 2
            - 1.0 / 6.0 * ((x + 1j * y) ** 3 - x ** 3)
            + 1.0 / 6.0 * x * (1j * y) ** 3
            + 0.25 * (x ** 2) * ((1j * y) ** 2)
            - 1.0 / 12.0 * (x ** 2) * ((1j * y) ** 3)
        )

    S = np.abs(E)

    fig, ax = pu.get_figure_and_axes(**figure_properties)
    pu.make_contourf(xv, yv, S, fig, ax, **contourf_properties)
    pu.set_axes_properties(ax, **axes_properties)
    pu.set_figure_properties(fig, **figure_properties)
    plt.show()
