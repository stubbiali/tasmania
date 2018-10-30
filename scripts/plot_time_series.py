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
import tasmania as taz


#
# User inputs
#
#root = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_centered_' \
#	   'nx51_ny51_nz50_dt20_nt1080_flat_terrain_L25000_u0_f_w6'
#root = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_centered_' \
#	   'nx51_ny51_nz50_dt15_nt5760_'
exts = ['cc_gaussian_L25000_H2250_u7.nc', 'sus_gaussian_L25000_H2250_u7.nc']

figsize = (7, 7)
fontsize = 16

linestyles = [
	'-',
	'--',
	':'
]

linecolors = [
	'black',
	'blue',
	'red',
]

linewidths = [
	1.0,
	1.5,
	1.5,
]

labels = [
	'CC',
	'SUS',
	'SSUS',
]

plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '',
	'title_right': '',
	'x_label': 'Elapsed time [h]',
	'x_lim': (0, 6),
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': range(0, 7),
	'x_ticklabels': range(0, 7),
	'xaxis_minor_ticks_visible': False,
	'xaxis_visible': True,
	'y_label': 'RRMSE [-]',
	'y_lim': None, #(0, 1),
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None,
	'y_ticklabels': None, #['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
	'yaxis_minor_ticks_visible': False,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	'legend_on': True,
	'legend_loc': 'best',
	'legend_framealpha': 1.0,
	'text': None,
	'text_loc': '',
	'grid_on': True,
	'grid_properties': {'linestyle': ':'},
}


#
# Code
#
if __name__ == '__main__':
	fig, ax = taz.get_figure_and_axes(figsize=figsize, fontsize=fontsize)

	grid, states_ref = taz.load_netcdf_dataset(root + exts[0])
	ns = grid.nx * grid.ny * grid.nz

	for k in range(1, len(exts)):
		fname = root + exts[k]
		grid, states = taz.load_netcdf_dataset(fname)

		t = []
		y = []

		for n in range(len(states)):
			su = states[n]['x_momentum_isentropic'].values
			s  = states[n]['air_isentropic_density'].values

			su_ref = states_ref[n]['x_momentum_isentropic'].values
			s_ref  = states_ref[n]['air_isentropic_density'].values

			t.append((states[n]['time'] - states[0]['time']).total_seconds() / 3600.0)
			y.append(np.linalg.norm(su / s - su_ref / s_ref) / np.linalg.norm(su_ref / s_ref))

		ax.plot(t, y, linestyle=linestyles[k], linewidth=linewidths[k],
				color=linecolors[k], label=labels[k])

		print('{} done.'.format(fname))

	taz.set_plot_properties(ax, **plot_properties)
	fig.tight_layout()

	plt.show()

