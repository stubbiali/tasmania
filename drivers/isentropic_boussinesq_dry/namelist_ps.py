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
from pandas import Timedelta
from sympl import DataArray

# backend settings
backend = gt.mode.NUMPY
dtype   = np.float64

# computational domain
domain_x = DataArray([-176, 176], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 41
domain_y = DataArray([-176, 176], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 41
domain_z = DataArray([340, 280], dims='potential_temperature', attrs={'units': 'K'})
nz       = 60

# horizontal boundary
hb_type = 'relaxed'
nb = 3
hb_kwargs = {'nr': 6}

# topography
topo_type   = 'gaussian'
topo_kwargs = {
	'time': timedelta(seconds=1800),
	'max_height': DataArray(1.0, attrs={'units': 'km'}),
	'width_x': DataArray(25.0, attrs={'units': 'km'}),
	'width_y': DataArray(25.0, attrs={'units': 'km'}),
	'smooth': False,
}

# initial conditions
init_time  = datetime(year=1992, month=2, day=20, hour=0)
x_velocity = DataArray(15.0, attrs={'units': 'm s^-1'})
y_velocity = DataArray(0.0, attrs={'units': 'm s^-1'})
brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

# time stepping
time_integration_scheme 		= 'rk3ws'
substeps                		= 0
physics_time_integration_scheme = 'rk2'

# advection
horizontal_flux_scheme = 'fifth_order_upwind'

# pressure gradient
pg_scheme = 'pressure_thickness_weighted'

# damping
damp                = True
damp_type           = 'rayleigh'
damp_depth          = 15
damp_max            = 0.0002
damp_at_every_stage = False

# horizontal diffusion
diff                = False
diff_type           = 'second_order'
diff_coeff          = DataArray(8800, attrs={'units': 's^-1'})
diff_coeff_max      = DataArray(8800, attrs={'units': 's^-1'})
diff_damp_depth     = 0

# horizontal smoothing
smooth                = False
smooth_type           = 'second_order'
smooth_coeff          = 1.0
smooth_coeff_max      = 1.0
smooth_damp_depth     = 0
smooth_at_every_stage = False

# turbulence
turbulence 			 = True
smagorinsky_constant = 0.18

# coriolis
coriolis           = True
coriolis_parameter = None  #DataArray(1e-3, attrs={'units': 'rad s^-1'})

# simulation length
timestep = Timedelta(seconds=18)
niter    = int(12*60*60 / timestep.total_seconds())

# output
filename = \
	'../../data/isentropic_dry_{}_{}_{}_pg{}_nx{}_ny{}_nz{}_dt{}_nt{}_' \
	'{}_L{}_H{}_u{}{}{}{}{}_ps.nc'.format(
		time_integration_scheme, horizontal_flux_scheme, physics_time_integration_scheme,
		2 if pg_scheme == 'second_order' else (4 if pg_scheme == 'fourth_order' else 'ptw'),
		nx, ny, nz, int(timestep.total_seconds()), niter,
		topo_type, int(topo_kwargs['width_x'].to_units('m').values.item()),
		int(topo_kwargs['max_height'].to_units('m').values.item()),
		int(x_velocity.to_units('m s^-1').values.item()),
		'_diff' if diff else '', '_smooth' if smooth else '',
		'_turb' if turbulence else '', '_f' if coriolis else ''
	)
filename = None
store_names = (
	'air_isentropic_density',
	'height_on_interface_levels',
	'x_momentum_isentropic',
	'y_momentum_isentropic'
)
save_frequency  = -1
print_frequency = 1
plot_frequency  = -1

