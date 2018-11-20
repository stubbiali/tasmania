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

# Backend settings
backend = gt.mode.NUMPY
dtype   = np.float64

# Computational domain
domain_x = DataArray([-250, 250], dims='x', attrs={'units': 'km'}).to_units('m')
nx       = 51
domain_y = DataArray([-250, 250], dims='y', attrs={'units': 'km'}).to_units('m')
ny       = 51
domain_z = DataArray([400, 300], dims='potential_temperature', attrs={'units': 'K'})
nz       = 50

# Topography
_width = DataArray(25.0, attrs={'units': 'km'})
topo_type   = 'gaussian'
topo_time   = timedelta(seconds=1800)
topo_kwargs = {
	#'topo_str': '1 * 10000. * 10000. / (x*x + 10000.*10000.)',
    #'topo_str': '3000. * pow(1. + (x*x + y*y) / 25000.*25000., -1.5)',
    'topo_max_height': DataArray(0.5, attrs={'units': 'km'}),
    'topo_width_x': _width,
    'topo_width_y': _width,
	'topo_smooth': False,
}

# Initial conditions
init_time       = datetime(year=1992, month=2, day=20, hour=0)
init_x_velocity = DataArray(1.0, attrs={'units': 'm s^-1'})
init_y_velocity = DataArray(0.0, attrs={'units': 'm s^-1'})
isothermal      = False
if isothermal:
    init_temperature = DataArray(250.0, attrs={'units': 'K'})
else:
    init_brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})

# Numerical scheme
time_integration_scheme  = 'rk3cosmo'
horizontal_flux_scheme   = 'fifth_order_upwind'
vertical_flux_scheme     = 'third_order_upwind'
horizontal_boundary_type = 'relaxed'

# Coupling
coupling_time_integration_scheme = 'rk2'

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
smooth_coeff          = 0.15
smooth_coeff_max      = 0.15
smooth_at_every_stage = False

# Prescribed surface heating
tendency_of_air_potential_temperature_in_diagnostics = True
amplitude_at_day_sw            		= DataArray(800.0, attrs={'units': 'W m^-2'})
amplitude_at_day_fw            		= DataArray(800.0, attrs={'units': 'W m^-2'})
amplitude_at_night_sw               = DataArray(-75.0, attrs={'units': 'W m^-2'})
amplitude_at_night_fw               = DataArray(-75.0, attrs={'units': 'W m^-2'})
frequency_sw						= DataArray(np.pi/12.0, attrs={'units': 'h^-1'})
frequency_fw						= DataArray(4.0*np.pi, attrs={'units': 'h^-1'})
attenuation_coefficient_at_day		= DataArray(1.0/600.0, attrs={'units': 'm^-1'})
attenuation_coefficient_at_night    = DataArray(1.0/75.0, attrs={'units': 'm^-1'})
characteristic_length               = DataArray(3.0 * _width.values.item(),
									 		    attrs={'units': _width.attrs['units']})
starting_time                       = init_time + timedelta(hours=8)

# Coriolis
coriolis_parameter = None

timestep = timedelta(seconds=10)
niter    = int(20*60*60 / timestep.total_seconds())

filename        = '../data/smolarkiewicz_{}_{}_{}_nx{}_ny{}_nz{}_dt{}_nt{}_{}_L{}_H{}_u{}_wf4_f_ssus.nc'.format(
					time_integration_scheme, horizontal_flux_scheme, vertical_flux_scheme,
					nx, ny, nz, int(timestep.total_seconds()), niter, topo_type,
					int(_width.to_units('m').values.item()),
					int(topo_kwargs['topo_max_height'].to_units('m').values.item()),
					int(init_x_velocity.to_units('m s^-1').values.item()))
save_frequency  = 180
print_frequency = 180
plot_frequency  = -1
