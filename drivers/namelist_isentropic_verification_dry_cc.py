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


dtype   = np.float32
backend = gt.mode.NUMPY

domain_x = DataArray([0, 500], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 51
domain_y = DataArray([-250, 250], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 51
domain_z = DataArray([400, 300], dims='potential_temperature', attrs={'units': 'K'})
nz       = 50

topo_type   = 'gaussian'
topo_time   = timedelta(seconds=1800)
topo_kwargs = {
	'topo_str': '1 * 10000. * 10000. / (x*x + 10000.*10000.)',
	'topo_max_height': DataArray(1000, attrs={'units': 'm'}),
	'topo_width_x': DataArray(50, attrs={'units': 'km'}),
	'topo_width_y': DataArray(50, attrs={'units': 'km'}),
	'topo_smooth': False,
}

init_time	 	 = datetime(year=1992, month=2, day=20)
init_x_velocity  = DataArray(15.0, attrs={'units': 'm s^-1'})
init_y_velocity  = DataArray(0.0, attrs={'units': 'm s^-1'})
isothermal 		 = False
if isothermal:
	init_temperature = DataArray(250.0, attrs={'units': 'K'})
else:
	init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

time_integration_scheme  = 'rk3cosmo'
horizontal_flux_scheme   = 'fifth_order_upwind'
horizontal_boundary_type = 'relaxed'

# Damping, i.e., wave absorber
damp_on             = True
damp_type           = 'rayleigh'
damp_depth          = 15
damp_max            = 0.0002
damp_at_every_stage = False

# Smoothing, i.e., digital filtering
smooth_on             = True
smooth_type           = 'second_order'
smooth_damp_depth     = 0
smooth_coeff          = 0.03
smooth_coeff_max      = 0.03
smooth_at_every_stage = False

timestep = timedelta(seconds=24)
niter    = 1800

filename        = '../data/verification_dry_{}_{}_nx{}_ny{}_nz{}_dt{}_nt{}_cc.nc'.format(
					time_integration_scheme, horizontal_flux_scheme, nx, ny, nz, 
					int(timestep.total_seconds()), niter)
save_frequency  = 60
print_frequency = 60
plot_frequency  = -1
