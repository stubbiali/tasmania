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


def test_animation_plot_1d(isentropic_moist_sedimentation_data):
	# Make sure the folder tests/baseline_images/test_animation does exist
	baseline_dir = 'baseline_images/test_animation'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the folder tests/result_images/test_animation does exist
	result_dir = 'result_images/test_animation'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Make sure the baseline image will exist at the end of this run
	filename = 'test_animation_plot_1d.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'precipitation'

	# Grab data
	grid, states = isentropic_moist_sedimentation_data

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'title_left': 'Precipitation [mm h$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 1.0],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}
		
	# Instantiate the monitor which generates the framework
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
				   fontsize=16, plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)

	# Create the animation
	from tasmania.plot.animation import Animation
	animation = Animation(monitor, fontsize=16, figsize=[7, 7], print_time='elapsed', fps=8)
	for state in states:
		animation.store(state)
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


def test_animation_plots_overlapper(isentropic_moist_sedimentation_data,
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
	filename = 'test_animation_plot_1d_overlapper.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	#
	# Plot1d#1
	#
	# Grab data
	grid, states1 = isentropic_moist_sedimentation_evaporation_data

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
		'legend_label': 'LF, evap. ON',
	}
	
	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, levels,
					interactive=False, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Grab data
	grid, states2 = isentropic_moist_sedimentation_data

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'green',
		'linestyle': '--',
		'linewidth': 1.5,
		'legend_label': 'MC, evap. OFF',
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# PlotsOverlapper
	#
	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'title_left': 'Accumulated precipitation [mm]',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 1.0],
		'grid_on': True,
		'legend_on': True,
		'legend_loc': 'best',
	}

	# Instantiate the artist which generates the frames
	from tasmania.plot.assemblers import PlotsOverlapper
	assembler = PlotsOverlapper([monitor1, monitor2], interactive=False,
							   plot_properties=plot_properties)

	# Create the animation
	from tasmania.plot.animation import Animation
	animation = Animation(assembler, fontsize=16, print_time='elapsed', fps=8)
	for state1, state2 in zip(states1, states2):
		animation.store([state1, state2])
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


def test_animation_plot_2d(isentropic_dry_data):
	# Make sure the folder tests/baseline_images/test_animation does exist
	baseline_dir = 'baseline_images/test_animation'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the folder tests/result_images/test_animation does exist
	result_dir = 'result_images/test_animation'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Make sure the baseline image will exist at the end of this run
	filename = 'test_animation_plot_2d.mp4'
	baseline_img = os.path.join(baseline_dir, filename)
	result_img = os.path.join(result_dir, filename)
	save_dest = result_img if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'horizontal_velocity'

	# Index identifying the cross-section to visualize
	z_level = -1

	# Load data
	grid, states = isentropic_dry_data

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'title_left': 'Horizontal velocity [m s$^{-1}$]',
		'x_label': '$x$ [km]',
		'y_label': '$y$ [km]',
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1e-3,
		'cmap_name': 'BuRd',
		'cbar_levels': 14,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 6.5,
	}
		
	# Instantiate the monitor which generates the frames
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xy import make_contourf_xy as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, z_level, interactive=False,
				   plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	# Create the animation
	from tasmania.plot.animation import Animation
	animation = Animation(monitor, fontsize=16, print_time='elapsed', fps=8)
	for state in states:
		animation.store(state)
	animation.run(save_dest=save_dest)

	# Asserts
	assert os.path.exists(save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
