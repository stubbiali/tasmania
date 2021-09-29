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
import matplotlib.pyplot as plt
import numpy as np
from tasmania.python.plot.plot_utils import (
    get_figure_and_axes,
    make_contourf,
    set_figure_properties,
)

a = np.random.rand(10)
x, y = np.meshgrid(sorted(a), sorted(a))
field = np.random.rand(10, 10)

fig, ax = get_figure_and_axes(
    nrows=2, ncols=4, index=1, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=2, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=3, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=4, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=5, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=6, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=7, figsize=(14, 8.75), fontsize=16
)
fig, _ = get_figure_and_axes(
    fig=fig, nrows=2, ncols=4, index=8, figsize=(14, 8.75), fontsize=16
)
make_contourf(
    x,
    y,
    field,
    fig,
    ax,
    cmap_name="viridis_white",
    cbar_on=True,
    cbar_levels=17,
    cbar_ticks_step=4,
    cbar_ticks_pos="interface",
    cbar_center=0.5,
    cbar_half_width=0.5,
    cbar_x_label="",
    cbar_y_label="$\\vert r \\, (\\beta \\Delta t, \\, \\alpha \\Delta t) \\vert$",
    cbar_title="",
    cbar_orientation="vertical",
    cbar_ax=[0, 1, 2, 3, 4, 5],
    cbar_format="%2.1f  ",
    cbar_extend="max",
    cbar_extendfrac="auto",
    cbar_extendrect=True,
    fontsize=16,
)
set_figure_properties(fig, tight_layout=True, fontsize=16)
ax.set_visible(False)
plt.show()
