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
from matplotlib.testing.decorators import image_comparison
import os
import pytest
import sys
sys.path.append(os.path.dirname(__file__))

from conftest import isentropic_dry_data, \
					 isentropic_moist_sedimentation_data, \
					 isentropic_moist_sedimentation_evaporation_data


@image_comparison(baseline_images=['test_plot_1d_x'], extensions=['eps'])
def test_plot_1d_x():
	# Make sure the folder tests/baseline_images/test_plots_assembler does exist
	baseline_dir = 'baseline_images/test_plots_overlapper'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_1d_x.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	#
	# Plot1d#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

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
	monitor1 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

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
		'title_left': '$y = ${} km, $\\theta = ${} K'.format(grid.y[0]/1e3, grid.z[-1]),
		'title_right': str(state1['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 1.0],
		'legend_on': True,
		'legend_loc': 'best',
		'grid_on': True,
	}

	# Plot
	from tasmania.plot.assemblers import PlotsOverlapper as Assembler
	assembler = Assembler([monitor1, monitor2], interactive=False,
						  figsize=[8, 8], plot_properties=plot_properties)
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_1d_z'], extensions=['eps'])
def test_plot_1d_z():
	# Make sure the folder tests/baseline_images/test_plots_assembler does exist
	baseline_dir = 'baseline_images/test_plots_overlapper'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_1d_z.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'mass_fraction_of_cloud_liquid_water_in_air'

	#
	# Plot1d#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {0: 40, 1: 0}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e3,
		'y_factor': 1.,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
		'legend_label': 'LF, evap. ON',
	}

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_vertical_profile as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-section to visualize
	levels = {0: 40, 1: 0}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e3,
		'y_factor': 1.,
		'linecolor': 'lightblue',
		'linestyle': '--',
		'linewidth': 1.5,
		'legend_label': 'MC, evap. OFF',
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# PlotAssembler
	#
	# Plot properties
	plot_properties = {
		'title_left': '$x = ${} km, $y = ${} km'.format(grid.x[40]/1e3, grid.y[0]/1e3),
		'title_right': str(state1['time'] - states[0]['time']),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.12],
		'y_label': '$\\theta$ [K]',
		'legend_on': True,
		'legend_loc': 'best',
		'grid_on': True,
	}

	# Plot
	from tasmania.plot.assemblers import PlotsOverlapper as Assembler
	assembler = Assembler([monitor1, monitor2], interactive=False,
						  figsize=[8, 8], plot_properties=plot_properties)
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_2d'], extensions=['eps'])
def test_plot_2d():
	# Make sure the folder tests/baseline_images/test_plots_assembler does exist
	baseline_dir = 'baseline_images/test_plots_overlapper'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_2d.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'horizontal_velocity'

	#
	# Plot2d#1
	#
	# Load data
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Index identifying the cross-section
	z_level = -1

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1e-3,
		'field_bias': 0.,
		'field_factor': 1.,
		'cmap_name': 'BuRd',
		'cbar_levels': 14,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 6.5,
		'cbar_x_label': '',
		'cbar_y_label': '',
		'cbar_title': '',
		'cbar_orientation': 'horizontal',
	}

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xy import make_contourf_xy as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, z_level, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# Plot2d#2
	#
	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'x_step': 2,
		'y_factor': 1e-3,
		'y_step': 2,
		'field_bias': 0.,
		'field_factor': 1.,
		'cmap_name': None,
	}

	# Instantiate the monitor
	from tasmania.plot.quiver_xy import make_quiver_xy as plot_function
	monitor2 = Plot(grid, plot_function, field_to_plot, z_level, interactive=False,
					plot_function_kwargs=plot_function_kwargs)

	#
	# PlotsOverlapper
	#
	# Plot properties
	plot_properties = {
		'title_left': 'Horizontal velocity [m s$^{-1}$] at the surface',
		'title_right': str(state1['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$y$ [km]',
		'y_lim': [-250, 250],
	}

	# Plot
	from tasmania.plot.assemblers import PlotsOverlapper as Assembler
	assembler = Assembler([monitor1, monitor2], interactive=False,
						  figsize=[7, 8], plot_properties=plot_properties)
	assembler.store([state1, state1], save_dest=save_dest, show=False)


if __name__ == '__main__':
	pytest.main([__file__])