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
import tasmania as taz


contourf_xz_kwargs = {
	'fontsize': 16,
	'x_factor': 1e-3,
	'z_factor': 1e-3,
	'field_bias': 10.,
	'field_factor': 1e4,
	'draw_grid': False,
	'cmap_name': 'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': 0,
	'cbar_half_width': 470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
}


plot_function_kwargs = {
	'contourf_xz': contourf_xz_kwargs,
}


plot_function = {
	'contourf_xz': taz.make_contourf_xz,
}


plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '$u\' = u - \\bar{u}$  [10$^{-4}$ m s$^{-1}$]',
	'title_right': '',
	'x_label': '$x$ [km]',
	'x_lim': [-40, 40],
	'x_ticks': None,
	'x_ticklabels': None,
	'xaxis_visible': True,
	'y_label': '$z$ [km]',
	'y_lim': [0, 15],
	'y_ticks': None,
	'y_ticklabels': None,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_visible': True,
	'legend_on': False,
	'legend_loc': 'best',
	'text': None,
	'text_loc': '',
	'grid_on': False,
}


if __name__ == '__main__':
	# User inputs
	filename = '../data/isentropic_convergence_rk3cosmo_fifth_order_upwind_nx51_dt20_nt6000.nc'
	time_level = -1
	field_to_plot = 'x_velocity_at_u_locations'
	level = 0
	plot_type = 'contourf_xz'
	fontsize = 16
	figsize = (7, 8)
	tight_layout = True

	# Load the data
	grid, states = taz.load_netcdf_dataset(filename)
	grid.update_topography(states[time_level]['time'] - states[0]['time'])
	state = states[time_level]

	# Instantiate the artist, then plot
	plot_properties['title_right'] = str(state['time'] - states[0]['time'])
	artist = taz.Plot2d(grid, plot_function[plot_type], field_to_plot, level,
						interactive=False, fontsize=16, figsize=figsize,
						tight_layout=tight_layout, plot_properties=plot_properties,
				   		plot_function_kwargs=plot_function_kwargs[plot_type])
	artist.store(state, show=True)
