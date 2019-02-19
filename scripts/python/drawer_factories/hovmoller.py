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


#==================================================
# User inputs
#==================================================
field_name  = 'x_velocity'
field_units = 'm s^-1'

x = 25
y = 25
z = None

axis_name  = 'height_on_interface_levels'
axis_units = 'km'
axis_x = None
axis_y = None
axis_z = None

time_mode  = 'elapsed'
init_time  = None
time_units = 'hr'

drawer_properties = {
	'fontsize': 16,
	'cmap_name': 'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': 15,
	'cbar_half_width': None, #8.5, #470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
	'draw_vertical_levels': False,
	'linecolor': 'black',
	'linewidth': 1.2,
}


#==================================================
# Code
#==================================================
def get_drawer(df_module):
	import_str = 'from {} import get_grid'.format(df_module)
	exec(import_str)
	grid = locals()['get_grid']()

	drawer = taz.HovmollerDiagram(
		grid, field_name, field_units, x=x, y=y, z=z,
		axis_name=axis_name, axis_units=axis_units,
		axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
		time_mode=time_mode, init_time=init_time,
		time_units=time_units, properties=drawer_properties
	)

	return drawer


def get_state(df_module, drawer, tlevel, axes_properties=None, print_time=None):
	import_str = 'from {} import get_state as df_get_state'.format(df_module)
	exec(import_str)

	tlevel = loader.nt + tlevel if tlevel < 0 else tlevel

	# assumption: any drawer is always called on the same dataset
	if drawer_tlevel >= tlevel:
		for k in range(tlevel, drawer_tlevel+1):
			drawer._data.pop(k)
	else:
		for k in range(drawer_tlevel+1, tlevel):
			drawer(locals()['df_get_state'](k))

	return locals()['df_get_state'](tlevel)
