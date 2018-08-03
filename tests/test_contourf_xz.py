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
import pickle
import pytest

from conftest import isentropic_dry_data, isentropic_moist_data


@image_comparison(baseline_images=['test_contourf_xz_velocity'], extensions=['eps'])
def test_contourf_xz_velocity():
	# Field to plot
	field_to_plot = 'x_velocity_at_u_locations'

	# Make sure the folder tests/baseline_images/test_contourf_xz does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_contourf_xz')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_contourf_xz_velocity.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	y_level = int(grid.ny/2)

	# Plot properties
	plot_properties = {
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'z_factor': 1e-3,
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
	
	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xz import make_contourf_xz as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_contourf_xz_isentropic_density'], extensions=['eps'])
def test_contourf_xz_isentropic_density():
	# Field to plot
	field_to_plot = 'air_isentropic_density'

	# Make sure the folder tests/baseline_images/test_contourf_xz does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_contourf_xz')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_contourf_xz_isentropic_density.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_moist_data()
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'title_left': 'Isentropic density [kg m$^{-2}$ K$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'UW',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'field_bias': 0.,
		'field_factor': 1.,
		'cmap_name': 'Blues',
		'cbar_levels': 14,
		'cbar_ticks_step': 4,
		'cbar_center': None,
		'cbar_half_width': None,
		'cbar_orientation': 'horizontal',
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xz import make_contourf_xz as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_contourf_xz_cloud_liquid_water'], extensions=['eps'])
def test_contourf_xz_cloud_liquid_water():
	# Field to plot
	field_to_plot = 'mass_fraction_of_cloud_liquid_water_in_air'

	# Make sure the folder tests/baseline_images/test_contourf_xz does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_contourf_xz')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_contourf_xz_cloud_liquid_water.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_moist_data()
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'title_left': 'Cloud liquid water [g kg$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'UW',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'field_bias': 0.,
		'field_factor': 1e3,
		'cmap_name': 'Blues',
		'cbar_levels': 14,
		'cbar_ticks_step': 4,
		'cbar_center': None,
		'cbar_half_width': None,
		'cbar_orientation': 'horizontal',
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xz import make_contourf_xz as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
