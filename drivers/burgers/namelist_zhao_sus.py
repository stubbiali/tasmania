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
import pandas as pd
from sympl import DataArray

factor = 3

# backend settings
backend = gt.mode.NUMPY
dtype   = np.float64

# computational domain
domain_x = DataArray([0, 1], dims='x', attrs={'units': 'm'})
nx       = 10*(2**factor) + 1
domain_y = DataArray([0, 1], dims='y', attrs={'units': 'm'})
ny       = 10*(2**factor) + 1

# initial conditions
init_time = datetime(year=1992, month=2, day=20, hour=0)

# numerical scheme
time_integration_scheme 		= 'rk3ws'
flux_scheme  					= 'fifth_order'
nb								= 3
physics_time_integration_scheme = 'rk2'

# diffusion
diffusion_type  = 'fourth_order'
diffusion_coeff = DataArray(0.1, attrs={'units': 'm^2 s^-1'})

# time
cfl      = 1.0
timestep = pd.Timedelta(cfl/(nx-1)**2, unit='s')
niter    = 4**factor * 100  #int(1 / timestep.total_seconds())

# output
filename = None
	#\
	#'../../data/burgers_{}_{}_{}_nx{}_ny{}_dt{}_nt{}_sus.nc'.format(
	#	time_integration_scheme, flux_scheme, physics_time_integration_scheme,
	#	nx, ny, int(timestep.total_seconds()), niter,
	#)
save_frequency  = -1
print_frequency = 100
plot_frequency  = -1