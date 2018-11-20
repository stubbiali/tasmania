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
from datetime import datetime
from loader import LoaderFactory
import tasmania as taz

#
# User inputs
#
filename1 = '../data/compressed/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
			'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_wf4_f_sus.nc'
filename2 = '../data/compressed/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
			'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_wf4_f_cc.nc'

field_name  = 'x_velocity_at_u_locations'
field_units = 'm s^-1'

x1, x2 = slice(3, -3, None), slice(3, -3, None)
y1, y2 = slice(3, -3, None), slice(3, -3, None)
z1, z2 = None, None

time_mode     = 'elapsed'
init_time     = datetime(year=1992, month=2, day=20, hour=0)
time_units    = 'hr'
time_on_xaxis = True

drawer_properties = {
	'fontsize': 16,
	'linestyle': '-',
	'linewidth': 1.5,
	'linecolor': 'green',
	'marker': '<',
	'markersize': 7,
	'markeredgewidth': 1,
	'markerfacecolor': 'green',
	'markeredgecolor': 'green',
	'legend_label': 'SUS'
}


#
# Code
#
def get_drawer():
	loader1 = LoaderFactory.factory(filename1)
	grid1 = loader1.get_grid()

	drawer = taz.TimeSeries(
		grid1, 'rrmsd_of_' + field_name, None,
		time_mode=time_mode, init_time=init_time,
		time_units=time_units, time_on_xaxis=time_on_xaxis,
		properties=drawer_properties
	)

	return drawer


def get_state(tlevel, drawer, axes_properties=None, print_time=None):
	loader1 = LoaderFactory.factory(filename1)
	loader2 = LoaderFactory.factory(filename2)

	grid1 = loader1.get_grid()
	grid2 = loader2.get_grid()

	rrmsd = taz.RRMSD(
		(grid1, grid2), {field_name: field_units},
		x=(x1, x2), y=(y1, y2), z=(z1, z2)
	)

	drawer.reset()

	tlevel = loader1.nt + tlevel if tlevel < 0 else tlevel

	for k in range(0, tlevel-1):
		state1, state2 = loader1.get_state(k), loader2.get_state(k)
		diagnostics = rrmsd(state1, state2)
		state1.update(diagnostics)
		drawer(state1)

	state1, state2 = loader1.get_state(tlevel), loader2.get_state(tlevel)
	diagnostics = rrmsd(state1, state2)
	state1.update(diagnostics)

	return state1

