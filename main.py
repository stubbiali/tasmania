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
import tasmania.utils.utils as utils
user_namelist = None if len(sys.argv) == 1 else sys.argv[1]
utils.set_namelist(user_namelist)

from tasmania.grids.grid_xyz import GridXYZ as Grid
from tasmania.dycore.dycore import DynamicalCore
from tasmania.model import Model
import tasmania.namelist as nl

#
# Instantiate the grid
#
print('\nCreate grid ...')
start = time.time()

grid = Grid(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
			units_x = 'm', dims_x = 'x', units_y = 'm', dims_y = 'y', 
			units_z = 'K', dims_z = 'air_potential_temperature', z_interface = None,
			topo_type = nl.topo_type, topo_time = nl.topo_time, **nl.topo_kwargs)

stop = time.time()
print('Grid created in {} ms.\n'.format((stop-start) * 1000.))

#
# Instantiate the dycore
#
print('Instantiate dycore ...')
start = time.time()

dycore = DynamicalCore.factory(nl.model, nl.time_scheme, nl.flux_scheme, nl.horizontal_boundary_type, grid, 
							   nl.moist_on, nl.backend, nl.damp_on, nl.damp_type, nl.damp_depth, nl.damp_max, 
				 		  	   nl.smooth_on, nl.smooth_type, nl.smooth_damp_depth, 
							   nl.smooth_coeff, nl.smooth_coeff_max, 
				 		  	   nl.smooth_moist_on, nl.smooth_moist_type, nl.smooth_moist_damp_depth, 
							   nl.smooth_moist_coeff, nl.smooth_moist_coeff_max,
							   nl.coupling_physics_dynamics_on, nl.sedimentation_on)

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
except EnvironmentError:
	pass
except TypeError:
	pass

