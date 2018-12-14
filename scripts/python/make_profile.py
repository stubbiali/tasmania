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
filename = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
		   'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_wf4_f_cc.nc'

field_name  = 'y_velocity_at_v_locations'
field_units = 'm s^-1'

x = 25
y = None
z = -1

axis_name  = 'x'
axis_units = 'km'
axis_x = None
axis_y = None
axis_z = None

drawer_properties = {
	'fontsize': 16,
	'linestyle': '-',
	'linewidth': 1.1,
	'linecolor': 'black',
	'marker': None,
	'markersize': None,
	'markeredgewidth': None,
	'markerfacecolor': None,
	'markeredgecolor': None,
	'legend_label': 'CC',
}


#
# Code
#
def get_drawer():
	loader = LoaderFactory.factory(filename)
	grid = loader.get_grid()

	drawer = taz.LineProfile(
		grid, field_name, field_units, x=x, y=y, z=z,
		axis_name=axis_name, axis_units=axis_units,
		axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
		properties=drawer_properties
	)

	return drawer


def get_state(tlevel, drawer=None, axes_properties=None, print_time=None):
	loader = LoaderFactory.factory(filename)
	state = loader.get_state(tlevel)

	if axes_properties is not None:
		if print_time == 'elapsed':
			init_time = loader.get_state(0)['time']
			axes_properties['title_right'] = str(state['time'] - init_time)
		elif print_time == 'absolute':
			axes_properties['title_right'] = str(state['time'])

	return state
