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
from datetime import timedelta
import gridtools as gt
import numpy as np
import os
import tasmania as taz

# Load the computational grid and the initial state
grid, states = taz.load_netcdf_dataset(
	'../tests/baseline_datasets/isentropic_moist_sedimentation.nc')
state = states[0]

# The component calculating the microphysical sources/sinks
# prescribed by the Kessler scheme; neglect the evaporation of
# rain in the subcloud layers
kessler = taz.Kessler(grid, pressure_on_interface_levels=True,
                      rain_evaporation_on=False, backend=gt.mode.NUMPY)

# The component performing the saturation adjustment
# as prescribed by the Kessler scheme
saturation = taz.SaturationAdjustmentKessler(grid, pressure_on_interface_levels=True,
                                             backend=gt.mode.NUMPY)

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=True,
                                     time_integration_scheme='forward_euler',
                                     horizontal_flux_scheme='upwind',
                                     horizontal_boundary_type='relaxed',
                                     smooth_on=True, smooth_type='first_order',
                                     smooth_coeff=0.20, smooth_at_every_stage=True,
                                     adiabatic_flow=True, sedimentation_on=False,
                                     backend=gt.mode.NUMPY, dtype=np.float64)

# The artist generating the left subplot
subplot1 = taz.Plot2d(grid, plot_function=taz.make_contourf_xz,
                      field_to_plot='x_velocity_at_u_locations', level=0,
                      plot_properties={'fontsize': 16,
                                       'title_left': '$x$-velocity [m s$^{-1}$]',
                                       'x_label': '$x$ [km]', 'x_lim': [0, 500],
                                       'y_label': '$z$ [km]', 'y_lim': [0, 14]},
                      plot_function_kwargs={'fontsize': 16,
                                            'x_factor': 1e-3, 'z_factor': 1e-3,
                                            'cmap_name': 'BuRd', 'cbar_on': True,
                                            'cbar_levels': 18, 'cbar_ticks_step': 2,
                                            'cbar_center': 15, 'cbar_half_width': 8.5,
                                            'cbar_orientation': 'horizontal'})

# The artist generating the right subplot
subplot2 = taz.Plot2d(grid, plot_function=taz.make_contourf_xz,
                      field_to_plot='mass_fraction_of_cloud_liquid_water_in_air', level=0,
                      plot_properties={'fontsize': 16,
                                       'title_left': 'Cloud liquid water [g kg$^{-1}$]',
                                       'x_label': '$x$ [km]', 'x_lim': [0, 500],
                                       'y_lim': [0, 14], 'yaxis_visible': False},
                      plot_function_kwargs={'fontsize': 16,
                                            'x_factor': 1e-3, 'z_factor': 1e-3,
                                            'field_factor': 1e3,
                                            'cmap_name': 'Blues', 'cbar_on': True,
                                            'cbar_levels': 18, 'cbar_ticks_step': 4,
                                            'cbar_orientation': 'horizontal'})

# The monitor encompassing and coordinating the two artists
monitor = taz.SubplotsAssembler(nrows=1, ncols=2, artists=(subplot1, subplot2),
                                interactive=True, figsize=(12, 7), fontsize=16,
                                tight_layout=True)

# Create a monitor to dump the solution into a NetCDF file
filename = '../tests/baseline_datasets/isentropic_moist.nc'
if os.path.exists(filename):
	os.remove(filename)
netcdf_monitor = taz.NetCDFMonitor(filename, grid)
netcdf_monitor.store(state)

# Simulation settings
timestep = timedelta(seconds=10)
niter = 2160

# Integrate
for i in range(niter):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * timestep)

	# Calculate the microphysical tendencies
	tendencies, _ = kessler(state)

	# Step the solution
	state_new = dycore(state, tendencies, timestep)
	state.update(state_new)

	# Perform the saturation adjustment
	state_new = saturation(state)
	state.update(state_new)

	if (i + 1) % 60 == 0:
		# Plot the solution
		subplot1.plot_properties['title_right'] = str((i + 1) * timestep)
		subplot2.plot_properties['title_right'] = str((i + 1) * timestep)
		fig = monitor.store((state, state))

		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')