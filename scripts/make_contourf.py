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
from loader import LoaderFactory
import tasmania as taz


#
# User inputs
#
plot_function_kwargs = {
	'fontsize': 16,
	'x_factor': 1e-3,
	'y_factor': 1e-3,
	'field_bias': 0., #10.,
	'field_factor': 1., #1e4,
	'cmap_name': 'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': None,
	'cbar_half_width': None, #8.5, #470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
}

plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': 'Horizontal velocity [m s$^{-1}$]', #''$u\' = u - \\bar{u}$  [10$^{-4}$ m s$^{-1}$]',
	'title_right': '',
	'x_label': '', #''$x$ [km]',
	'x_lim': [-200, 200],
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': None,
	'x_ticklabels': None,
	'xaxis_minor_ticks_visible': True,
	'xaxis_visible': True,
	'y_label': '', #''$y$ [km]',
	'y_lim': [-200, 200],
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None,
	'y_ticklabels': None,
	'yaxis_minor_ticks_visible': True,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	'legend_on': False,
	'legend_loc': 'best',
	'text': None,
	'text_loc': '',
	'grid_on': False,
	'grid_properties': {'linestyle': ':'},
}

filename = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_centered_nx51_ny51_nz50_' \
	       'dt20_nt2160_flat_terrain_L25000_u0_f_w1_cc.nc'
time_level = -3
field_to_plot = 'horizontal_velocity'
level = -1
fontsize = 16
figsize = (7, 8)
tight_layout = True
print_time = None # 'elapsed', 'absolute'


#
# Code
#
def get_artist(tlevel=None):
	tlevel = tlevel if tlevel is not None else time_level

	# Load the data
	grid, states = taz.load_netcdf_dataset(filename)
	grid.update_topography(states[tlevel]['time'] - states[0]['time'])
	state = states[tlevel]

	# Print time
	if print_time == 'elapsed':
		plot_properties['title_right'] = str(state['time'] - states[0]['time'])
	elif print_time == 'absolute':
		plot_properties['title_right'] = str(state['time'])

	# Instantiate the artist
	artist = taz.Plot2d(grid, taz.make_contourf_xy, field_to_plot, level,
						interactive=False, fontsize=16, figsize=figsize,
						tight_layout=tight_layout, plot_properties=plot_properties,
						plot_function_kwargs=plot_function_kwargs)

	return artist, state


def get_loader():
	return LoaderFactory(filename)


if __name__ == '__main__':
	artist, state = get_artist()
	artist.store(state, show=True)
