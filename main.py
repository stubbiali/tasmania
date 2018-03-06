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
"""
Script for building and simulating a three-dimensional moist isentropic model, according 
to the model, physical and numerical settings specified in the :mod:`namelist` module.

Warning
-------
Do not modify this file, unless you know what you are doing!
"""
from datetime import datetime, timedelta
import os
import pickle
import sys
import time

# Set namelist
import utils.utils as utils
user_namelist = None if len(sys.argv) == 1 else sys.argv[1]
utils.set_namelist(user_namelist)

from grids.grid_xyz import GridXYZ as Grid
from dycore.dycore import DynamicalCore
from model import Model
import namelist as nl

#
# Instantiate the grid
#
print('\nCreate grid ...')
start = time.time()

grid = Grid(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
			units_x = 'm', dims_x = 'x', units_y = 'm', dims_y = 'y', 
			units_z = 'K', dims_z = 'potential_temperature', z_interface = None,
			topo_type = nl.topo_type, topo_time = nl.topo_time, **nl.topo_kwargs)

stop = time.time()
print('Grid created in {} ms.\n'.format((stop-start) * 1000.))

#
# Instantiate the dycore
#
print('Instantiate dycore ...')
start = time.time()

dycore = DynamicalCore.factory(nl.model, nl.time_scheme, nl.flux_scheme, nl.horizontal_boundary_type, grid, 
							   nl.imoist, nl.backend, nl.idamp, nl.damp_type, nl.damp_depth, nl.damp_max, 
				 		  	   nl.ismooth, nl.smooth_type, nl.smooth_damp_depth, 
							   nl.smooth_coeff, nl.smooth_coeff_max, 
				 		  	   nl.ismooth_moist, nl.smooth_moist_type, nl.smooth_moist_damp_depth, 
							   nl.smooth_moist_coeff, nl.smooth_moist_coeff_max)

stop = time.time()
print('Dycore instantiated in {} ms.\n'.format((stop-start) * 1000.))

#
# Instantiate the diagnostics
#
# ...

#
# Instantiate the model
#
print('Instantiate model ...')
start = time.time()

model = Model(dycore)

stop = time.time()
print('Model instantiated in {} ms.\n'.format((stop-start) * 1000.))

#
# Get the initial state
#
print('Compute initial state ...')
start = time.time()

state = dycore.get_initial_state(nl.initial_time, nl.initial_state_type, **nl.initial_state_kwargs)

stop = time.time()
print('Initial state computed in {} ms.\n'.format((stop-start) * 1000.))

#
# Integrate
#
print('Start simulation ...\n')
start = time.time()

state_out, state_save = model(nl.dt, nl.simulation_time, state, save_iterations = nl.save_iterations)

stop = time.time()
print('\nSimulation completed in {} s.\n'.format(stop-start))

#
# Save to output
#
try:
	with open(nl.save_dest, 'wb') as output:
		pickle.dump(state_save, output)
except EnvironmentError, TypeError:
	pass
	pass
