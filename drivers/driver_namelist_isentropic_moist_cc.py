import os
import tasmania as taz

import namelist_isentropic_moist_cc as nl


# Create the underlying grid
grid = taz.GridXYZ(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
				   topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
				   dtype=nl.dtype)

# Instantiate the initial state
state = taz.get_default_isentropic_state(grid, nl.init_time,
										 nl.init_x_velocity, nl.init_y_velocity,
										 nl.init_brunt_vaisala, moist_on=True, dtype=nl.dtype)

# Instantiate the component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.NonconservativeIsentropicPressureGradient(grid, order=order,
                                                   horizontal_boundary_type=nl.horizontal_boundary_type, 
                                                   backend=nl.backend, dtype=nl.dtype)

# Instantiate the component implementing the Kessler microphysics scheme
ks = taz.Kessler(grid, pressure_on_interface_levels=True, 
                 tendency_of_air_potential_temperature_in_diagnostics=True,
                 rain_evaporation=nl.rain_evaporation, backend=nl.backend)

# Wrap the components calculating tendencies in a ConcurrentCoupling object
cc = taz.ConcurrentCoupling(pg, ks, mode='as_parallel')

# Instantiate the component retrieving the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=True, pt=pt,
                               backend=nl.backend, dtype=nl.dtype)

# The component performing the saturation adjustment 
# as prescribed by the Kessler scheme
sat = taz.SaturationAdjustmentKessler(grid, pressure_on_interface_levels=True, backend=nl.backend)

# Wrap the diagnostic components in a DiagnosticComponentComposite object
dcc = taz.DiagnosticComponentComposite(dv, sat)

# Instatiate the dynamical core
dycore = taz.HomogeneousIsentropicDynamicalCore(grid, moist_on=True,
                                                # Numerical scheme
                                                time_integration_scheme=nl.time_integration_scheme,
                                                horizontal_flux_scheme=nl.horizontal_flux_scheme,
                                                horizontal_boundary_type=nl.horizontal_boundary_type,
                                                # Parameterizations
                                                intermediate_parameterizations=cc,
                                                diagnostics=dcc,
                                                # Damping (wave absorber)
                                                damp_on=nl.damp_on, damp_type=nl.damp_type,
                                                damp_depth=nl.damp_depth, damp_max=nl.damp_max,
                                                damp_at_every_stage=nl.damp_at_every_stage,
                                                # Smoothing
                                                smooth_on=nl.smooth_on, smooth_type=nl.smooth_type,
                                                smooth_coeff=nl.smooth_coeff, 
                                                smooth_coeff_max=nl.smooth_coeff_max,
                                                smooth_at_every_stage=nl.smooth_at_every_stage,
                                                # Smoothing on water species
                                                smooth_moist_on=nl.smooth_moist_on, 
                                                smooth_moist_type=nl.smooth_moist_type,
                                                smooth_moist_coeff=nl.smooth_moist_coeff,
                                                smooth_moist_coeff_max=nl.smooth_moist_coeff_max,
                                                smooth_moist_at_every_stage=nl.smooth_moist_at_every_stage,
                                                # Implementation details
                                                backend=nl.backend, dtype=nl.dtype)

# Create a monitor to dump to the solution into a NetCDF file
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

	# Step the solution
	state_new = dycore(state, {}, dt)
	state.update(state_new)

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
