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

from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.monitors import Plot


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_profile')
def test_profile_x(isentropic_moist_sedimentation_data):
	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	# Make sure the folder tests/baseline_images/test_profile does exist
	baseline_dir = 'baseline_images/test_profile'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_x_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z,
						 axis_units='km', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$y = ${} km, $\\theta = ${} K'.format(
			grid.y.values[0]/1e3, grid.z.values[-1]),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_profile')
def test_profile_y(isentropic_dry_data):
	# Field to plot
	field_name  = 'y_velocity_at_v_locations'
	field_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_profile does exist
	baseline_dir = 'baseline_images/test_profile'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_y_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	x, z = int(grid.nx/2), -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, x=x, z=z,
						 axis_units='km', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x = ${} km, $\\theta = ${} K'.format(
			grid.x.values[int(grid.nx/2)]/1e3, grid.z.values[-1]),
		'x_label': '$y$ [km]',
		'x_lim': None,
		'y_label': '$y$-velocity [m s$^{-1}$]',
		'y_lim': [-4.5, 4.5],
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_profile')
def test_profile_z(isentropic_moist_sedimentation_data):
	# Field to plot
	field_name  = 'mass_fraction_of_cloud_liquid_water_in_air'
	field_units = 'g kg^-1'

	# Make sure the folder tests/baseline_images/test_profile does exist
	baseline_dir = 'baseline_images/test_profile'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_z_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	x, y = 40, 0

	# Drawer properties
	drawer_properties = {
		'linecolor': 'lightblue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, x=x, y=y,
						 axis_units='K', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x = ${} km, $y = ${} km'.format(
			grid.x.values[40]/1e3, grid.y.values[0]/1e3),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.2],
		'y_label': '$\\theta$ [K]',
		'y_lim': None,
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_profile')
def test_profile_h(isentropic_moist_sedimentation_data):
	# Field to plot
	field_name  = 'mass_fraction_of_cloud_liquid_water_in_air'
	field_units = 'g kg^-1'

	# Make sure the folder tests/baseline_images/test_profile does exist
	baseline_dir = 'baseline_images/test_profile'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_h_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	x, y = 40, 0

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'linecolor': 'lightblue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, x=x, y=y,
						 axis_name='height', axis_units='km', properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x = ${} km, $y = ${} km'.format(
			grid.x.values[40]/1e3, grid.y.values[0]/1e3),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.2],
		'y_label': '$z$ [km]',
		'y_lim': None,
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_profile_x_transformation(isentropic_dry_data())
