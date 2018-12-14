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
import os
import pytest

from tasmania.plot.monitors import Plot
from tasmania.plot.quiver import Quiver


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_quiver')
def test_quiver_xy_velocity(isentropic_dry_data, drawer_topography2d):
	# Field to plot
	xcomp_name  = 'x_velocity'
	xcomp_units = 'm s^-1'
	ycomp_name  = 'y_velocity'
	ycomp_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_quiver does exist
	baseline_dir = 'baseline_images/test_quiver'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_quiver_xy_velocity_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'x_step': 2,
		'y_step': 2,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 14,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 6.5,
		'cbar_x_label': '',
		'cbar_y_label': '',
		'cbar_title': '',
		'cbar_orientation': 'horizontal',
		'alpha': 0.5,
	}

	# Instantiate the drawer
	drawer = Quiver(grid, z=z, xcomp_name=xcomp_name, xcomp_units=xcomp_units,
					ycomp_name=ycomp_name, ycomp_units=ycomp_units,
					xaxis_units='km', yaxis_units='km', properties=drawer_properties)

	# Instantiate the drawer plotting the topography
	topo_drawer = drawer_topography2d(grid, xaxis_units='km', yaxis_units='km')

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Surface horizontal velocity [m s$^{-1}$]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
	}

	# Instantiate the monitor
	monitor = Plot((topo_drawer, drawer), False, figure_properties, axes_properties)

	# Plot
	monitor.store((state, state), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_quiver')
def test_quiver_xy_velocity_bw(isentropic_dry_data, drawer_topography2d):
	# Field to plot
	xcomp_name  = 'x_velocity'
	xcomp_units = 'm s^-1'
	ycomp_name  = 'y_velocity'
	ycomp_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_quiver does exist
	baseline_dir = 'baseline_images/test_quiver'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_quiver_xy_velocity_bw_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'x_step': 2,
		'y_step': 2,
		'alpha': 0.5,
	}

	# Instantiate the drawer
	drawer = Quiver(grid, z=z, xcomp_name=xcomp_name, xcomp_units=xcomp_units,
					ycomp_name=ycomp_name, ycomp_units=ycomp_units,
					xaxis_units='km', yaxis_units='km', properties=drawer_properties)

	# Instantiate the drawer plotting the topography
	topo_drawer = drawer_topography2d(grid, xaxis_units='km', yaxis_units='km')

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Surface horizontal velocity [m s$^{-1}$]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
	}

	# Instantiate the monitor
	monitor = Plot((topo_drawer, drawer), False, figure_properties, axes_properties)

	# Plot
	monitor.store((state, state), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_quiver_xy_velocity(isentropic_dry_data())