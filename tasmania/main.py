"""
Script for building and simulating a three-dimensional moist isentropic model, according 
to the model, physical and numerical settings specified in the :mod:`~tasmania.namelist` module.

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
from set_namelist import set_namelist
user_namelist = None if len(sys.argv) == 1 else sys.argv[1]
set_namelist(user_namelist)

from tasmania.grids.grid_xyz import GridXYZ as Grid
from tasmania.dycore.dycore import DynamicalCore
from tasmania.model import Model
import tasmania.namelist as nl
from tasmania.parameterizations.adjustments import AdjustmentMicrophysics
from tasmania.parameterizations.slow_tendencies import SlowTendencyMicrophysics

#
# Instantiate the grid
#
print('\nCreate the grid ...')
start = time.time()

grid = Grid(domain_x    = nl.domain_x, 
			nx          = nl.nx, 
			domain_y    = nl.domain_y, 
			ny          = nl.ny, 
			domain_z    = nl.domain_z, 
			nz          = nl.nz,
			units_x     = 'm', 
			dims_x      = 'x', 
			units_y     = 'm', 
			dims_y      = 'y', 
			units_z     = 'K', 
			dims_z      = 'air_potential_temperature', 
			z_interface = None,
			topo_type   = nl.topo_type, 
			topo_time   = nl.topo_time, 
			**nl.topo_kwargs)

stop = time.time()
print('Grid created in %5.5f ms.\n' % ((stop-start) * 1000.))

#
# Instantiate the model
#
print('Instantiate the model ...')
start = time.time()

model = Model()

stop = time.time()
print('Model instantiated in %5.5f ms.\n' % ((stop-start) * 1000.))

#
# Instantiate the dycore, then add it to the model
#
print('Instantiate the dycore ...')
start = time.time()

dycore = DynamicalCore.factory(model                        = nl.model, 
							   time_scheme                  = nl.time_scheme, 
							   flux_scheme                  = nl.flux_scheme, 
							   horizontal_boundary_type     = nl.horizontal_boundary_type, 
							   grid                         = grid, 
							   moist_on                     = nl.moist_on, 
							   backend                      = nl.backend, 
							   damp_on                      = nl.damp_on, 
							   damp_type                    = nl.damp_type, 
							   damp_depth                   = nl.damp_depth, 
							   damp_max                     = nl.damp_max, 
				 		  	   smooth_on                    = nl.smooth_on, 
							   smooth_type                  = nl.smooth_type, 
							   smooth_damp_depth            = nl.smooth_damp_depth, 
							   smooth_coeff                 = nl.smooth_coeff, 
							   smooth_coeff_max             = nl.smooth_coeff_max, 
				 		  	   smooth_moist_on              = nl.smooth_moist_on, 
							   smooth_moist_type            = nl.smooth_moist_type, 
							   smooth_moist_damp_depth      = nl.smooth_moist_damp_depth, 
							   smooth_moist_coeff           = nl.smooth_moist_coeff, 
							   smooth_moist_coeff_max       = nl.smooth_moist_coeff_max,
							   physics_dynamics_coupling_on = nl.physics_dynamics_coupling_on, 
							   sedimentation_on             = nl.sedimentation_on,
							   sedimentation_flux_type      = nl.sedimentation_flux_type,
							   sedimentation_substeps       = nl.sedimentation_substeps)
model.set_dynamical_core(dycore)

stop = time.time()
print('Dycore instantiated in %5.5f ms.\n' % ((stop-start) * 1000.))

#
# Instantiate the parameterization calculating slow-varying cloud microphysics tendencies, 
# then add it to the model
#
if nl.slow_tendency_microphysics_on:
	print('Instantiate the class calculating slow-varying cloud microphysics tendencies ...')
	start = time.time()

	slow_tendency_microphysics = SlowTendencyMicrophysics.factory(micro_scheme        = nl.slow_tendency_microphysics_type, 
													 		 	  grid                = grid, 
													 		 	  rain_evaporation_on = nl.rain_evaporation_on, 
													 		 	  backend             = nl.backend, 
													 		 	  **nl.slow_tendency_microphysics_kwargs)
	model.add_slow_tendency(slow_tendency_microphysics)

	stop = time.time()
	print('Class calculating slow-varying cloud microphysics tendencies instantiated in %5.5f ms.\n' %
		  ((stop-start) * 1000.))

#
# Instantiate the parameterization performing cloud microphysical adjustments, 
# then add it to the model
#
if nl.adjustment_microphysics_on:
	print('Instantiate the class performing cloud microphysics adjustments ...')
	start = time.time()

	adjustment_microphysics = AdjustmentMicrophysics.factory(micro_scheme        = nl.adjustment_microphysics_type, 
													 		 grid                = grid, 
													 		 rain_evaporation_on = nl.rain_evaporation_on, 
													 		 backend             = nl.backend, 
													 		 **nl.adjustment_microphysics_kwargs)
	model.add_adjustment(adjustment_microphysics)

	stop = time.time()
	print('Class performing cloud microphysics adjustments instantiated in %5.5f ms.\n' %
		  ((stop-start) * 1000.))

#
# Get the initial state
#
print('Compute the initial state ...')
start = time.time()

state = dycore.get_initial_state(initial_time       = nl.initial_time, 
								 initial_state_type = nl.initial_state_type, 
								 **nl.initial_state_kwargs)

stop = time.time()
print('Initial state computed in %5.5f ms.\n' % ((stop-start) * 1000.))

#
# Run the simulation
#
print('Start the simulation ...\n')
start = time.time()

state_out, state_save, diagnostics_save = model(dt              = nl.dt, 
							  					simulation_time = nl.simulation_time, 
							  					state           = state, 
							  					save_iterations = nl.save_iterations)

stop = time.time()
print('\nSimulation completed in %5.5f s.\n' % (stop-start))

#
# Save
#
try:
	with open(nl.save_dest, 'wb') as output:
		pickle.dump(state_save, output)
		pickle.dump(diagnostics_save, output)
except EnvironmentError:
	print('Data have not been saved due to an EnvironmentError.')
except TypeError:
	print('Data have not been saved due to a TypeError.')

