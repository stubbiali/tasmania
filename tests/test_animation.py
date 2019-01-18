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

from tasmania.python.plot.animation import Animation
from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.monitors import Plot, PlotComposite
from tasmania.python.plot.profile import LineProfile


def test_profile(isentropic_moist_sedimentation_data):
	# Make sure the folder tests/baseline_images/test_animation does exist
	baseline_dir = 'baseline_images/test_animation'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the folder tests/result_images/test_animation does exist
	result_dir = 'result_images/test_animation'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Make sure the baseline image will exist at the end of this run
	filename = 'test_profile.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_name  = 'precipitation'
	field_units = 'mm h^-1'

	# Grab data
	grid, states = isentropic_moist_sedimentation_data

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
		'title_left': 'Precipitation [mm h$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 1.0],
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Create the animation
	animation = Animation(monitor, print_time='elapsed', fps=8)
	for state in states:
		animation.store(state)
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


def test_plot_composite(isentropic_moist_sedimentation_data,
						isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_animation does exist
	baseline_dir = 'baseline_images/test_animation'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the folder tests/result_images/test_animation does exist
	result_dir = 'result_images/test_animation'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Make sure the baseline image will exist at the end of this run
	filename = 'test_plot_composite.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	#
	# Plot#1
	#
	# Load data
	grid, states1 = isentropic_moist_sedimentation_evaporation_data

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

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Leapfrog',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states2 = isentropic_moist_sedimentation_data

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z,
						 axis_units='km', properties=drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (8, 7),
		'tight_layout': True,
	}

	# Instantiate the monitor
	monitor = PlotComposite(1, 2, (plot1, plot2), False, figure_properties)

	# Create the animation
	animation = Animation(monitor, print_time='elapsed', fps=8)
	for state1, state2 in zip(states1, states2):
		animation.store([state1, state2])
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


def test_contourf(isentropic_dry_data, drawer_topography2d):
	# Make sure the folder tests/baseline_images/test_animation does exist
	baseline_dir = 'baseline_images/test_animation'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the folder tests/result_images/test_animation does exist
	result_dir = 'result_images/test_animation'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Make sure the baseline image will exist at the end of this run
	filename = 'test_contourf.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_name  = 'horizontal_velocity'
	field_units = 'm s^-1'

	# Grab data from dataset
	grid, states = isentropic_dry_data

	# Index identifying the cross-section to visualize
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'cmap_name': 'BuRd',
		'cbar_levels': 18,
		'cbar_ticks_step': 4,
		'cbar_center': 15.0,
		'cbar_orientation': 'horizontal',
		'alpha': 0.1,
		'colors': 'black',
	}

	# Instantiate the drawer
	drawer = Contourf(grid, field_name, field_units, z=z,
					  xaxis_units='km', yaxis_units='km',
					  properties=drawer_properties)

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
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
	}

	# Instantiate the monitor
	monitor = Plot((topo_drawer, drawer), False, figure_properties, axes_properties)

	# Create the animation
	animation = Animation(monitor, print_time='elapsed', fps=8)
	for state in states:
		grid.update_topography(state['time'] - states[0]['time'])
		animation.store((state, state))
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
