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

domain_x = DataArray([-250, 250], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 51
domain_y = DataArray([-250, 250], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 51
domain_z = DataArray([400, 300], dims='potential_temperature', attrs={'units': 'K'})
nz       = 60

topo_type   = 'gaussian'
topo_time   = timedelta(seconds=1800)
topo_kwargs = {
    'topo_max_height': DataArray(2.0, attrs={'units': 'km'}),
    'topo_width_x': DataArray(50.0, attrs={'units': 'km'}),
    'topo_width_y': DataArray(50.0, attrs={'units': 'km'}),
	'topo_smooth': False,
}

init_time          = datetime(year=1992, month=2, day=20)
init_x_velocity    = DataArray(15.0, attrs={'units': 'm s^-1'})
init_y_velocity    = DataArray(0.0, attrs={'units': 'm s^-1'})
init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

time_integration_scheme  = 'rk3cosmo'
horizontal_flux_scheme   = 'fifth_order_upwind'
horizontal_boundary_type = 'relaxed'

damp_on             = True
damp_type           = 'rayleigh'
damp_depth          = 15
damp_max            = 0.0002
damp_at_every_stage = False

smooth_on             = True
smooth_type           = 'third_order'
smooth_damp_depth     = 0
smooth_coeff          = 0.24
smooth_coeff_max      = 0.24
smooth_at_every_stage = False

smooth_moist_on             = False
smooth_moist_type           = 'third_order'
smooth_moist_damp_depth     = 30
smooth_moist_coeff          = 0.2
smooth_moist_coeff_max      = 1.0
smooth_moist_at_every_stage = True

sedimentation    = False
rain_evaporation = True

timestep = timedelta(seconds=30)
niter    = int(21600 / timestep.total_seconds())

filename        = None #'../data/isentropic_convergence_{}_{}.nc'.format(horizontal_flux_scheme, nx)
save_frequency  = -1
print_frequency = -1
plot_frequency  = 10
