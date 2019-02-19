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
from sympl import DataArray
from loader import LoaderFactory
import tasmania as taz

from tasmania.python.utils.dict_utils import subtract


#==================================================
# User inputs
#==================================================
filename = '../data/isentropic_convergence_rk3cosmo_fifth_order_upwind_nx801_dt1_nt96000.nc'
x_velocity_initial = DataArray(10.0, attrs={'units': 'm s^-1'})
temperature = DataArray(250.0, attrs={'units': 'K'})
mountain_height = DataArray(1.0, attrs={'units': 'm'})
mountain_width = DataArray(10.0, attrs={'units': 'km'})

field_units = 'mm das^-1'

x = None
y = 0
z = None

xaxis_name  = 'x'
xaxis_units = 'km'
xaxis_y = None
xaxis_z = None

yaxis_name  = 'y'
yaxis_units = 'km'
yaxis_x = None
yaxis_z = None

zaxis_name  = 'height'
zaxis_units = 'km'
zaxis_x = None
zaxis_y = None

drawer_properties = {
	'fontsize': 16,
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
	'draw_vertical_levels': False,
}


#==================================================
# Code
#==================================================
def get_grid():
	loader = LoaderFactory.factory(filename)
	return loader.get_grid()


def get_state(tlevel):
	loader = LoaderFactory.factory(filename)
	grid = loader.get_grid()
	init_state = loader.get_state(0)

	u, _ = taz.get_isothermal_isentropic_analytical_solution(
		grid, x_velocity_initial, temperature, mountain_height, mountain_width
	)
	final_state = {'x_velocity_at_u_locations': u}

	state = subtract(final_state, init_state, unshared_variables_in_output=False)
	state.update({
		key: value for key, value in init_state.items()
		if key != 'x_velocity_at_u_locations'
	})

	return state


def get_initial_time():
	loader = LoaderFactory.factory(filename)
	return loader.get_state(0)['time']
