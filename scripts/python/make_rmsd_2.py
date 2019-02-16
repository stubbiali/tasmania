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

#==================================================
# User inputs
#==================================================
filename1 = '../../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx81_ny81_nz60_dt24_nt1800_ns0_flat_terrain_L50000_H0_u0_wf3_f_lazy.nc'
filename2 = '../../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx81_ny81_nz60_dt24_nt1800_ns0_flat_terrain_L50000_H0_u0_wf3_f_cc.nc'

field_name  = 'x_momentum_isentropic'
field_units = 'kg m^-1 K^-1 s^-1'

x1, x2 = slice(20, 61, None), slice(20, 61, None)
y1, y2 = slice(20, 61, None), slice(20, 61, None)
z1, z2 = slice(0, 60, None), slice(0, 60, None)

time_mode     = 'elapsed'
init_time     = datetime(year=1992, month=2, day=20, hour=0)
time_units    = 'hr'
time_on_xaxis = True

drawer_properties = {
	'fontsize': 16,
	'linestyle': '-',
	'linewidth': 1.5,
	'linecolor': 'green',
	'marker': 's',
	'markersize': 7,
	'markeredgewidth': 1,
	'markerfacecolor': 'green',
	'markeredgecolor': 'green',
	'legend_label': 'LCC'
}


#==================================================
# Code
#==================================================
def get_drawer():
	loader1 = LoaderFactory.factory(filename1)
	grid1 = loader1.get_grid()

	drawer = taz.TimeSeries(
		grid1, 'rmsd_of_' + field_name, None,
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

	rmsd = taz.RMSD(
		(grid1, grid2), {field_name: field_units},
		x=(x1, x2), y=(y1, y2), z=(z1, z2)
	)

	drawer.reset()

	tlevel = loader1.nt + tlevel if tlevel < 0 else tlevel

	for k in range(0, tlevel-1):
		state1, state2 = loader1.get_state(k), loader2.get_state(k)
		diagnostics = rmsd(state1, state2)
		state1.update(diagnostics)
		drawer(state1)

	state1, state2 = loader1.get_state(tlevel), loader2.get_state(tlevel)
	diagnostics = rmsd(state1, state2)
	state1.update(diagnostics)

	return state1

