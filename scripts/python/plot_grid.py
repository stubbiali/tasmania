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
from sympl import DataArray
import tasmania as taz
from python.grids.sigma import Sigma3d

domain_x = DataArray([-200, 200], dims="x", attrs={"units": "km"}).to_units(
    "m"
)
nx = 101
domain_y = DataArray([-1, 1], dims="y", attrs={"units": "km"}).to_units("m")
ny = 1
domain_z = DataArray(
    [0.1, 1], dims="potential_temperature", attrs={"units": "1"}
)
nz = 40

z_interface = None  # DataArray(0.3, attrs={'units': '1'})

topo_type = "gaussian"
topo_kwargs = {
    "topo_max_height": DataArray(2, attrs={"units": "km"}),
    "topo_width_x": DataArray(25, attrs={"units": "km"}),
}

figure_properties = {
    "fontsize": 16,
    "figsize": (6, 6),
    "tight_layout": True,
    "tight_layout_rect": None,  # (0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
    "fontsize": 16,
    "title_center": "",
    "title_left": "",
    "title_right": "",
    "x_label": "$x$ [km]",  #'Time (UTC)',
    "x_lim": (-200, 200),  # (-190, 210),
    "invert_xaxis": False,
    "x_scale": None,
    "x_ticks": range(-200, 201, 100),  # (-190, -90, 10, 110, 210),
    "x_ticklabels": None,
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    "y_label": "$z$ [km]",
    "y_lim": (0, 15),  # (-200, 200), # (0.01, 0.15),
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": range(0, 16, 3),
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
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper right",
    "grid_on": False,
    "grid_properties": {"linestyle": ":"},
}

if __name__ == "__main__":
    grid = Sigma3d(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        z_interface,
        topo_type=topo_type,
        topo_kwargs=topo_kwargs,
    )

    fig, ax = taz.get_figure_and_axes(**figure_properties)

    x = grid.x.to_units("km").values
    h = grid.height_on_interface_levels.to_units("km").values

    for k in range(0, nz + 1):
        ax.plot(x, h[:, 0, k], color="gray", linewidth=1, alpha=1)
        # ax.plot(x, h[0, 0, k]*np.ones(nx), color='gray', linewidth=1, alpha=1, zorder=1)
    ax.plot(x, h[:, 0, -1], color="black", linewidth=1.2, alpha=1)

    plt.fill_between(x, h[:, 0, -1], color="darkgrey", alpha=1, zorder=2)

    # ax.plot(x, 9*np.ones(nx), color='black', linewidth=1.2, linestyle='--', alpha=1)
    # plt.text(202, 9, '$z_F$', fontsize=16,
    # 		 horizontalalignment='left', verticalalignment='center')

    taz.set_axes_properties(ax, **axes_properties)
    taz.set_figure_properties(fig, **figure_properties)

    plt.show()
