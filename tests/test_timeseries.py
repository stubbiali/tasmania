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
import numpy as np
import os
import pytest
from sympl import DiagnosticComponent

from tasmania.python.plot.trackers import TimeSeries
from tasmania.python.plot.monitors import Plot


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_timeseries')
def test_datapoint(isentropic_dry_data):
	# Field to plot
	field_name  = 'x_velocity_at_u_locations'
	field_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_timeseries does exist
	baseline_dir = 'baseline_images/test_timeseries'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_datapoint_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data

	# Indices identifying the grid point to visualize
	x, y, z = 25, 25, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
		'marker': 'o',
	}

	# Instantiate the drawer
	drawer = TimeSeries(grid, field_name, field_units, x=x, y=y, z=z,
						time_mode='elapsed', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': False,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x$-velocity at $x = $0 km, $y = $0 km'.format(
			grid.z.values[-1]),
		'x_label': 'Elapsed time [s]',
		'y_label': '$\\theta$ [K]',
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	for state in states[:-1]:
		monitor.store(state)
	monitor.store(states[-1], save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_timeseries')
def test_diagnostic(isentropic_dry_data):
	class MaxVelocity(DiagnosticComponent):
		def __init__(self, grid):
			self.g = grid
			super().__init__()

		@property
		def input_properties(self):
			dims = (self.g.x_at_u_locations.dims[0], self.g.y.dims[0], self.g.z.dims[0])
			return {'x_velocity_at_u_locations': {'dims': dims, 'units': 'm s^-1'}}

		@property
		def diagnostic_properties(self):
			dims = ('scalar', 'scalar', 'scalar')
			return {'max_x_velocity_at_u_locations': {'dims': dims, 'units': 'm s^-1'}}

		def array_call(self, state):
			val = np.max(state['x_velocity_at_u_locations'])
			return {'max_x_velocity_at_u_locations':
						np.array(val)[np.newaxis, np.newaxis, np.newaxis]}

	# Field to plot
	field_name  = 'max_x_velocity_at_u_locations'
	field_units = 'km hr^-1'

	# Make sure the folder tests/baseline_images/test_timeseries does exist
	baseline_dir = 'baseline_images/test_timeseries'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_diagnostic_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data

	# Drawer properties
	drawer_properties = {
		'linecolor': 'green',
		'linestyle': None,
		'linewidth': 1.5,
		'marker': '^',
		'markersize': 2,
	}

	# Instantiate the drawer
	drawer = TimeSeries(grid, field_name, field_units,
						time_mode='elapsed', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'x_label': 'Elapsed time [s]',
		'y_label': 'Max. $x$-velocity [km h$^{-1}$]',
		'grid_on': True,
		'grid_properties': {'linestyle': ':'},
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Instantiate the diagnostic
	mv = MaxVelocity(grid)

	# Plot
	for state in states[:-1]:
		state.update(mv(state))
		monitor.store(state)
	states[-1].update(mv(states[-1]))
	monitor.store(states[-1], save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_datapoint(isentropic_dry_data())
