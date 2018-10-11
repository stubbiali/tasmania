import os
import tasmania as taz

import namelist_isentropic_dry_ssus as nl


# Create the underlying grid
grid = taz.GridXYZ(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
				   topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
				   dtype=nl.dtype)

# Instantiate the initial state
if nl.isothermal_flow:
	state = taz.get_isothermal_isentropic_state(grid, nl.init_time,
												nl.init_x_velocity, nl.init_y_velocity,
												nl.init_temperature, dtype=nl.dtype)
else:
	state = taz.get_default_isentropic_state(grid, nl.init_time
											 nl.init_x_velocity, nl.init_y_velocity,
											 nl.init_brunt_vaisala, moist_on=False, dtype=nl.dtype)

# Instantiate the component inferring the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt,
							   backend=nl.backend, dtype=nl.dtype)

# Instantiate the component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.ConservativeIsentropicPressureGradient(grid, order=order,
												horizontal_boundary_type=nl.horizontal_boundary_type,
												backend=nl.backend, dtype=nl.dtype)

# Instantiate the component retrieving the velocity components
vc = taz.IsentropicVelocityComponents(grid, horizontal_boundary_type=nl.horizontal_boundary_type,
									  reference_state=state, backend=nl.backend, dtype=nl.dtype)

# Wrap the physical components in a SequentialUpdateSplitting object
sus_bd = taz.SequentialUpdateSplitting(pg, vc, time_integration_scheme='forward_euler',
									   grid=grid, horizontal_boundary_type=None)
sus_ad = taz.SequentialUpdateSplitting(dv, pg, vc, time_integration_scheme='forward_euler',
									   grid=grid, horizontal_boundary_type=None)

# Instantiate the dry isentropic dynamical core
dycore = taz.HomogeneousIsentropicDynamicalCore(grid, moist_on=False,
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

# Initialize auxiliary flags
to_print = to_save = True

# Integrate
for i in range(nt):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	if to_print or to_save:
		# Compute the physics before the dynamics
		_ = sus_bd(state=state, timestep=0.5*timestep)

	# Calculate the dynamics
	state_new = dycore(state, {}, timestep)
	state.update(state_new)

	# Update the auxiliary flags
	to_print = (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0)
	to_save  = (nl.filename is not None) and \
			   (((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt)

	if to_print or to_save:
		# Calculate the physics, and couple it with the dynamics
		_ = sus_ad(state=state, timestep=0.5*timestep)
	else:
		# Fuse two consecutive calls to sus_ad/sus_bd into a single call to sus_ad
		_ = sus_ad(state=state, timestep=timestep)

	if (i+1) % niter == 0:
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

	if to_print:
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print('Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			  'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(i+1, cfl, umax, umin, vmax, vmin))

	if to_save:
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
if nl.filename is not None:
	netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
