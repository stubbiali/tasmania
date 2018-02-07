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
Script for building up and running a three-dimensional moist isentropic model, according 
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
import utils
user_namelist = None if len(sys.argv) == 1 else sys.argv[1]
utils.set_namelist(user_namelist)

from grids.xyz_grid import XYZGrid as Grid
from dycore.isentropic_dycore import IsentropicDynamicalCore
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
			topo_type = nl.topo_type, topo_time = nl.topo_time, 
			topo_max_height = nl.topo_max_height, 
			topo_width_x = nl.topo_width_x, topo_width_y = nl.topo_width_y)

stop = time.time()
print('Grid created in {} ms.\n'.format((stop-start) * 1000.))

#
# Instantiate the dycore
#
print('Instantiate dycore ...')
start = time.time()

dycore = IsentropicDynamicalCore(grid, nl.imoist, nl.horizontal_boundary_type, nl.scheme, nl.backend,
				 		  		 nl.idamp, nl.damp_type, nl.damp_depth, nl.damp_max, 
				 		  		 nl.idiff, nl.diff_coeff, nl.diff_coeff_moist, nl.diff_max)

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

state = dycore.get_initial_state(nl.initial_time, nl.x_velocity_initial, nl.y_velocity_initial, 
								 nl.brunt_vaisala_initial)

stop = time.time()
print('Initial state computed in {} ms.\n'.format((stop-start) * 1000.))

#
# Integrate
#
print('Start simulation ...\n')
start = time.time()

state_out, state_save = model(nl.dt, nl.simulation_time, state, save_freq = nl.save_freq)

stop = time.time()
print('\nSimulation completed in {} s.\n'.format(stop-start))

#
# Save to output
#
if nl.save_freq > 0:
	with open(nl.save_dest, 'wb') as output:
		pickle.dump(state_save, output)

