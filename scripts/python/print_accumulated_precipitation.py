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
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq5_turb_sed_evap_cc_sat.nc',
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq0_turb_sed_evap_lcc.nc',
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq5_turb_sed_evap_ps_sat.nc',
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq5_turb_sed_evap_sts_sat.nc',
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq5_turb_sed_evap_sus_sat.nc',
		'../../data/isentropic_moist_rk3ws_si_fifth_order_upwind_rk2_pg2_nx41_ny41_nz60_'
  			'dt40_nt270_gaussian_L50000_H1000_u15_rh90_thetas280_mcfreq5_turb_sed_evap_ssus_sat.nc',
	],
]
xslices = [slice(8, 33), ]*6
yslices = [slice(8, 33), ]*6
area_factors = [8800.0**2, ]*6 

# ==================================================
# Code
# ==================================================
for i in range(len(filenames)):
    for j in range(len(filenames[i])):
        domain, grid_type, states = taz.load_netcdf_dataset(filenames[i][j])
        state = states[-1]
        accprec = state['accumulated_precipitation'].to_units('m').values[xslices[i], yslices[i], 0]
        print(area_factors[i] * np.sum(np.sum(accprec, axis=1), axis=0))

