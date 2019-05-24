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
import tasmania as taz

# ==================================================
# User inputs
# ==================================================
filenames = [
	[
		'../../data/isentropic_moist_rk3ws_fifth_order_upwind_pg2_'
		'nx41_ny41_nz60_dt32_nt900_gaussian_L25000_H1000_u15_rh80_turb_f_sed_cc.nc',
		'../../data/isentropic_moist_rk3ws_fifth_order_upwind_rk2_pg2_'
		'nx41_ny41_nz60_dt32_nt900_gaussian_L25000_H1000_u15_rh80_turb_f_sed_ps.nc',
		'../../data/isentropic_moist_rk3ws_fifth_order_upwind_rk2_pg2_'
		'nx41_ny41_nz60_dt32_nt900_gaussian_L25000_H1000_u15_rh80_turb_f_sed_sus.nc',
		'../../data/isentropic_moist_rk3ws_fifth_order_upwind_rk2_pg2_'
		'nx41_ny41_nz60_dt32_nt900_gaussian_L25000_H1000_u15_rh80_turb_f_sed_ssus.nc',
	],
]
xslices = [slice(10, 31), slice(20, 61), slice(40, 121)]
yslices = [slice(10, 31), slice(20, 61), slice(40, 121)]
area_factors = [1e-6*8800.0**2, 1e-6*4400.0**2, 1e-6*2200.0**2] 

# ==================================================
# Code
# ==================================================
for i in range(len(filenames)):
    for j in range(len(filenames[i])):
        domain, grid_type, states = taz.load_netcdf_dataset(filenames[i][j])
        state = states[-1]
        accprec = state['accumulated_precipitation'].to_units('mm').values[xslices[i], yslices[i], 0]
        print(area_factors[i] * np.sum(np.sum(accprec, axis=1), axis=0))

