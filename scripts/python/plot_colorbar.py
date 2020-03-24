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
from tasmania.python.plot.plot_utils import (
    get_figure_and_axes,
    make_contourf,
    set_figure_properties,
)

a = np.random.rand(10)
x, y = np.meshgrid(sorted(a), sorted(a))
field = np.random.rand(10, 10)

fig, ax = get_figure_and_axes(nrows=1, ncols=1, index=1, figsize=(8, 2.5), fontsize=16)
make_contourf(
    x,
    y,
    field,
    fig,
    ax,
    cmap_name="RdBu_r",
    cbar_on=True,
    cbar_levels=26,
    cbar_ticks_step=6,
    cbar_ticks_pos="center",
    cbar_center=22.5,
    cbar_half_width=12.5,
    cbar_extend=True,
    cbar_orientation="horizontal",
    cbar_x_label="$x$-velocity [m s$^{-1}$]",
    fontsize=16
)
set_figure_properties(fig, tight_layout=True, fontsize=16)
ax.set_visible(False)
plt.show()
