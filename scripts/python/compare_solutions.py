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


#
# User inputs
#
filename1 = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
			'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_wf4_f_sus_1.nc'
filename2 = '../data/smolarkiewicz_rk3cosmo_fifth_order_upwind_third_order_upwind_' \
			'nx51_ny51_nz50_dt10_nt7200_gaussian_L25000_H500_u1_wf4_f_ssus.nc'

tlevels1 = range(18, 19)
tlevels2 = range(18, 19)

fieldname1 = 'y_velocity_at_v_locations'
fieldname2 = 'y_velocity_at_v_locations'

units1 = 'm s^-1'
units2 = 'm s^-1'

#
# Code
#
if __name__ == '__main__':
	grid1, states1 = taz.load_netcdf_dataset(filename1)
	grid2, states2 = taz.load_netcdf_dataset(filename2)

	for t1, t2 in zip(tlevels1, tlevels2):
		field1 = states1[t1][fieldname1].to_units(units1).values
		field2 = states2[t2][fieldname2].to_units(units2).values

		isclose = np.allclose(field1, field2)

		if isclose:
			print('Iteration ({:4d}, {:4d}) validated.'.format(t1, t2))
		else:
			print('Iteration ({:4d}, {:4d}) not validated.'.format(t1, t2))
