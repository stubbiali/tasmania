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
import numpy as np
import os
import tasmania as taz
from sympl import DataArray


#==================================================
# User inputs
#==================================================
filename_from = '../../data/isentropic_dry_rk3cosmo_fifth_order_upwind_' \
	'nx161_ny161_nz60_dt12_nt3600_gaussian_L50000_H1000_u15_f_cc.nc'
filename_to = '../../data/compressed/isentropic_dry_rk3cosmo_fifth_order_upwind_' \
	'nx161_ny161_nz60_dt12_nt3600_gaussian_L50000_H1000_u15_f_cc.nc'

field_names = [
	'height_on_interface_levels',
	'x_momentum_isentropic',
]

x = slice(60, 101, None)
y = slice(60, 101, None)
z = None

tlevels = range(0, 72)

dtype = np.float64


#==================================================
# Code
#==================================================
if __name__ == '__main__':
	grid, states_from = taz.load_netcdf_dataset(filename_from)

	x = x if x is not None else slice(0, grid.nx)
	domain_x = DataArray(
		[
			grid.x.values[x.start if x is not None else 0], 
			grid.x.values[x.stop-1 if x is not None else -1]
		], 
		dims=grid.x.dims[0], attrs={'units': grid.x.attrs['units']}
	)
	nx = x.stop - x.start if x is not None else grid.nx

	y = y if y is not None else slice(0, grid.ny)
	domain_y = DataArray(
		[
			grid.y.values[y.start if y is not None else 0], 
			grid.y.values[y.stop-1 if y is not None else -1]
		], 
		dims=grid.y.dims[0], attrs={'units': grid.y.attrs['units']}
	)
	ny = y.stop - y.start if y is not None else grid.ny

	z = z if z is not None else slice(0, grid.nz)
	z_hl = slice(z.start, z.stop+1)
	domain_z = DataArray(
		[
			grid.z_on_interface_levels.values[z.start if z is not None else 0], 
			grid.z_on_interface_levels.values[z.stop if z is not None else -1]
		], 
		dims=grid.z.dims[0], attrs={'units': grid.z.attrs['units']}
	)
	nz = z.stop - z.start if z is not None else grid.nz

	r_grid = taz.GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz, dtype=dtype)

	if os.path.exists(filename_to):
		os.remove(filename_to)
	netcdf_monitor = taz.NetCDFMonitor(filename_to, r_grid)

	for k in tlevels:
		state_from = states_from[k]

		state_to = {'time': state_from['time']}
		for name in field_names:
			if state_from[name].shape[2] == grid.nz: 
				state_to[name] = taz.make_data_array_3d(
					state_from[name].values[x, y, z], r_grid, units=state_from[name].attrs['units']
				)
			else:
				state_to[name] = taz.make_data_array_3d(
					state_from[name].values[x, y, z_hl], r_grid, units=state_from[name].attrs['units']
				)

		netcdf_monitor.store(state_to)

	netcdf_monitor.write()
