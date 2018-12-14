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
import os
import tasmania as taz


#
# User inputs
#
filename_from = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_f_ssus_1.nc'
filename_to = '../data/compressed/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
	'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_f_ssus_1.nc'

field_names = [
	'height_on_interface_levels',
	'x_velocity_at_u_locations',
	'y_velocity_at_v_locations',
]

tlevels = range(0, 41)


#
# Code
#
if __name__ == '__main__':
	grid, states_from = taz.load_netcdf_dataset(filename_from)

	if os.path.exists(filename_to):
		os.remove(filename_to)
	netcdf_monitor = taz.NetCDFMonitor(filename_to, grid)

	for k in tlevels:
		state_from = states_from[k]

		state_to = {'time': state_from['time']}
		for name in field_names:
			state_to[name] = state_from[name]

		netcdf_monitor.store(state_to)

	netcdf_monitor.write()
