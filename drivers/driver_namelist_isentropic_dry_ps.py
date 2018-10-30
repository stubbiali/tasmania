import os
import tasmania as taz

import namelist_isentropic_dry as nl


# Create the underlying grid
grid = taz.GridXYZ(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
				   topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
				   dtype=nl.dtype)

# Instantiate the initial state
state = taz.get_isothermal_isentropic_state(grid, nl.init_time,
											nl.init_x_velocity, nl.init_y_velocity,
											nl.init_temperature, dtype=nl.dtype)

# Instantiate the component inferring the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt,
							   backend=nl.backend, dtype=nl.dtype)

# Instantiate the component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.ConservativeIsentropicPressureGradient(grid, order=order,
	horizontal_boundary_type=nl.horizontal_boundary_type, backend=nl.backend, dtype=nl.dtype)

# Instantiate the component retrieving the velocity components
vc = taz.IsentropicVelocityComponents(grid, horizontal_boundary_type=nl.horizontal_boundary_type,
									  reference_state=state, backend=nl.backend, dtype=nl.dtype)

# Wrap the physical components in a ParallelSplitting object
ps = taz.ParallelSplitting(dv, pg, vc, mode='serial',
						   time_integration_scheme=nl.time_integration_scheme,
						   retrieve_diagnostics_from_provisional_state=True)

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=False,
									 time_integration_scheme=nl.time_integration_scheme,
									 horizontal_flux_scheme=nl.horizontal_flux_scheme,
									 horizontal_boundary_type=nl.horizontal_boundary_type,
									 damp_on=nl.damp_on, damp_type=nl.damp_type,
									 damp_depth=nl.damp_depth, damp_max=nl.damp_max,
									 damp_at_every_stage=nl.damp_at_every_stage,
									 smooth_on=nl.smooth_on, smooth_type=nl.smooth_type,
									 smooth_coeff=nl.smooth_coeff,
									 smooth_at_every_stage=nl.smooth_at_every_stage,
									 backend=nl.backend, dtype=nl.dtype)

# Create a monitor to dump the solution into a NetCDF file
if nl.filename is not None:
	if os.path.exists(nl.filename):
		os.remove(nl.filename)
	netcdf_monitor = taz.NetCDFMonitor(nl.filename, grid)
	netcdf_monitor.store(state)

# Simulation settings
dt = nl.timestep
nt = nl.niter

# Integrate
for i in range(nt):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	# Compute the dynamics
	state_prv = dycore(state, {}, dt)

	# Compute the physics, and couple it with the dynamics
	_ = ps(state=state, state_prv=state_prv, timestep=dt)

	# Update the state
	state.update(state_prv)

	if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print('Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			  'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(i+1, cfl, umax, umin, vmax, vmin))

	if (nl.filename is not None) and \
		(((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt):
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
if nl.filename is not None:
	netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
