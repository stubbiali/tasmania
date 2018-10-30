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
from datetime import datetime, timedelta
import gridtools as gt
import numpy as np
from sympl import DataArray


dtype   = np.float64
backend = gt.mode.NUMPY

domain_x = DataArray([-220, 220], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 101
domain_y = DataArray([-1, 1], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 1
domain_z = DataArray([765, 300], dims='potential_temperature', attrs={'units': 'K'})
nz       = 300

topo_type   = 'user_defined'
topo_time   = timedelta(seconds=900)
topo_kwargs = {
	'topo_str': '1 * 10000. * 10000. / (x*x + 10000.*10000.)',
	'topo_smooth': False,
}

init_time	 	 = datetime(year=1992, month=2, day=20)
init_x_velocity  = DataArray(10.0, attrs={'units': 'm s^-1'})
init_y_velocity  = DataArray(0.0, attrs={'units': 'm s^-1'})
isothermic_flow  = True
if isothermic_flow:
	init_temperature = DataArray(250.0, attrs={'units': 'K'})
else:
	init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

time_integration_scheme  = 'rk3cosmo'
horizontal_flux_scheme   = 'fifth_order_upwind'
horizontal_boundary_type = 'relaxed'

damp_on             = True
damp_type           = 'rayleigh'
damp_depth          = 150
damp_max            = 0.05
damp_at_every_stage = False

smooth_on             = False
smooth_type           = 'first'
smooth_coeff          = 0.05
smooth_at_every_stage = False

timestep = timedelta(seconds=10)
niter    = int(120000 / timestep.total_seconds())

filename        = '../data/isentropic_convergence_ssus_{}_{}_nx{}_dt{}_nt{}.nc'.format(
					time_integration_scheme, horizontal_flux_scheme, nx, int(timestep.total_seconds()), niter)
save_frequency  = int(niter/2)
print_frequency = 100

