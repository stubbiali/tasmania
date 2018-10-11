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
grid, states = taz.load_netcdf_dataset('../tests/baseline_datasets/isentropic_dry.nc')
state = states[0]

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=False,
                                     time_integration_scheme='forward_euler',
                                     horizontal_flux_scheme='maccormack',
                                     horizontal_boundary_type='relaxed',
                                     damp_on=True, damp_type='rayleigh', damp_depth=15,
                                     damp_max=0.0002, damp_at_every_stage=False,
                                     smooth_on=True, smooth_type='first_order',
                                     smooth_coeff=0.12, smooth_at_every_stage=False,
                                     backend=gt.mode.NUMPY, dtype=np.float32)

# The artist and its collaborators generating the left subplot
coll1 = taz.Plot2d(grid, plot_function=taz.make_contourf_xy,
                   field_to_plot='horizontal_velocity', level=-1,
                   plot_function_kwargs={'fontsize': 16,
                                         'x_factor': 1e-3, 'y_factor': 1e-3,
                                         'cmap_name': 'BuRd', 'cbar_on': True,
                                         'cbar_levels': 14, 'cbar_ticks_step': 2,
                                         'cbar_center': 15, 'cbar_half_width': 6.5,
                                         'cbar_orientation': 'horizontal',
                                         'cbar_x_label': 'Horizontal velocity [m s$^{-1}$]'})
coll2 = taz.Plot2d(grid, plot_function=taz.make_quiver_xy,
                   field_to_plot='horizontal_velocity', level=-1,
                   plot_function_kwargs={'fontsize': 16,
                                         'x_factor': 1e-3, 'x_step': 2,
                                         'y_factor': 1e-3, 'y_step': 2})
subplot1 = taz.PlotsOverlapper((coll1, coll2), fontsize=16,
                               plot_properties={'fontsize': 16,
                                                'title_left': '$\\theta = 300$ K',
                                                'x_label': '$x$ [km]', 'x_lim': [0, 500],
                                                'y_label': '$y$ [km]', 'y_lim': [-250, 250]})

# The artist generating the right subplot
subplot2 = taz.Plot2d(grid, plot_function=taz.make_contourf_xz,
                      field_to_plot='x_velocity_at_u_locations', level=25, fontsize=16,
                      plot_properties={'fontsize': 16, 'title_left': '$y = 0$ km',
                                       'x_label': '$x$ [km]', 'x_lim': [0, 500],
                                       'y_label': '$z$ [km]', 'y_lim': [0, 14]},
                      plot_function_kwargs={'fontsize': 16,
                                            'x_factor': 1e-3, 'z_factor': 1e-3,
                                            'cmap_name': 'BuRd', 'cbar_on': True,
                                            'cbar_levels': 14, 'cbar_ticks_step': 2,
                                            'cbar_center': 15, 'cbar_half_width': 6.5,
                                            'cbar_orientation': 'horizontal',
                                            'cbar_x_label': '$x$-velocity [m s$^{-1}$]'})

# The monitor encompassing and coordinating the two artists
monitor = taz.SubplotsAssembler(nrows=1, ncols=2, artists=(subplot1, subplot2),
                                interactive=True, figsize=(12, 7), fontsize=16,
                                tight_layout=True)

# Create a monitor to dump the solution into a NetCDF file
filename = '../tests/baseline_datasets/isentropic_dry.nc'
if os.path.exists(filename):
	os.remove(filename)
netcdf_monitor = taz.NetCDFMonitor(filename, grid)
netcdf_monitor.store(state)

# Simulation settings
timestep = timedelta(seconds=24)
niter = 1800

# Integrate
for i in range(niter):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * timestep)

	# Step the solution
	state_new = dycore(state, {}, timestep)
	state.update(state_new)

	if (i + 1) % 60 == 0:
		# Plot the solution
		subplot1.plot_properties['title_right'] = str((i + 1) * timestep)
		subplot2.plot_properties['title_right'] = str((i + 1) * timestep)
		fig = monitor.store(((state, state), state))

		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')