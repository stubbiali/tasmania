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


# Backend details
dtype   = np.float32
backend = gt.mode.NUMPY

domain_x = DataArray([-352, 352], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 41
domain_y = DataArray([-352, 352], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 41
domain_z = DataArray([340, 280], dims='potential_temperature', attrs={'units': 'K'})
nz       = 60

topo_type   = 'gaussian'
topo_time   = timedelta(seconds=1800)
topo_kwargs = {
	'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
	'topo_width_x': DataArray(50.0, attrs={'units': 'km'}),
	'topo_width_y': DataArray(50.0, attrs={'units': 'km'}),
	'topo_smooth': False,
}

init_time	 	   = datetime(year=1992, month=2, day=20)
init_x_velocity    = DataArray(15.0, attrs={'units': 'm s^-1'})
init_y_velocity    = DataArray(0.0, attrs={'units': 'm s^-1'})
isothermal		   = False
if isothermal:
	init_temperature = DataArray(300, attrs={'units': 'K'})
else:
	init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

time_integration_scheme  = 'rk3cosmo'
horizontal_flux_scheme   = 'fifth_order_upwind'
vertical_flux_scheme	 = 'third_order_upwind'
horizontal_boundary_type = 'relaxed'
substeps		 		 = 0

damp                = True			# False
damp_type           = 'rayleigh'	# 'rayleigh'
damp_depth          = 15			# 20
damp_max            = 0.0002			# 0.05
damp_at_every_stage = False			# True

smooth                = True			# True
smooth_type           = 'second_order'	# 'first_order'
smooth_damp_depth     = 0				# 30
smooth_coeff          = 0.03				# 0.2
smooth_coeff_max      = 0.03				# 1.0
smooth_at_every_stage = False			# True

smooth_moist                = False			# False
smooth_moist_type           = 'third_order'	# 'third_order'
smooth_moist_damp_depth     = 30			# 30
smooth_moist_coeff          = 0.2			# 0.2
smooth_moist_coeff_max      = 1.0			# 1.0
smooth_moist_at_every_stage = True			# True

precipitation    = False
rain_evaporation = False

coriolis_parameter = None

timestep = timedelta(seconds=24)
niter    = 1800  #int(12*60*60 / timestep.total_seconds())

filename        = \
	'../../data/isentropic_dry_{}_{}_{}_nx{}_ny{}_nz{}_dt{}_nt{}_ns{}_cc.nc'.format(
		time_integration_scheme, horizontal_flux_scheme, vertical_flux_scheme,
		nx, ny, nz, int(timestep.total_seconds()), niter, substeps
	)
save_frequency  = 60
print_frequency = 60
