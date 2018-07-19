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
from matplotlib.testing.decorators import image_comparison
import os
import pickle
import pytest


@image_comparison(baseline_images=['test_plot_1d'], extensions=['eps'])
def test_plot_1d():
	# Make sure the folder tests/baseline_images/test_subplots_composer does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_subplots_assembler')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_1d.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	#
	# Plot1d#1
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state1 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'Leapfrog',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
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
	
	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist_maccormack.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state2 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 1.0],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# SubplotsAssembler
	#
	# Plot
	from tasmania.plot.assemblers import SubplotsAssembler
	assembler = SubplotsAssembler(1, 2, [monitor1, monitor2],
								  interactive=False, fontsize=16, figsize=[8, 7])
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_1d_share_yaxis'], extensions=['eps'])
def test_plot_1d_share_yaxis():
	# Make sure the folder tests/baseline_images/test_subplots_composer does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_subplots_assembler')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_1d_share_yaxis.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	#
	# Plot1d#1
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state1 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'Leapfrog',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
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

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist_maccormack.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state2 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 1.0],
		'y_ticklabels': [],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# SubplotsAssembler
	#
	# Plot
	from tasmania.plot.assemblers import SubplotsAssembler
	assembler = SubplotsAssembler(1, 2, [monitor1, monitor2],
								  interactive=False, fontsize=16, figsize=[8, 7])
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_1d_share_xaxis'], extensions=['eps'])
def test_plot_1d_share_xaxis():
	# Make sure the folder tests/baseline_images/test_subplots_composer does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_subplots_assembler')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_1d_share_xaxis.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	#
	# Plot1d#1
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state1 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'Leapfrog',
		'x_lim': [0, 500],
		'x_ticklabels': [],
		'y_label': 'Accumulated precipitation [mm]',
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

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot1d#2
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist_maccormack.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state2 = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 1.0],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# SubplotsAssembler
	#
	# Plot
	from tasmania.plot.assemblers import SubplotsAssembler
	assembler = SubplotsAssembler(2, 1, [monitor1, monitor2],
								  interactive=False, fontsize=16, figsize=[7, 8])
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_2d_one_row_two_columns'], extensions=['eps'])
def test_plot_2d_one_row_two_columns():
	# Make sure the folder tests/baseline_images/test_subplots_assembler does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_subplots_assembler')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_2d_one_row_two_columns.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'x_velocity_at_u_locations'

	#
	# Plot2d#1
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state1 = states[-1]

	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Horizontal velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BuRd',
		'cbar_on': False,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 9.5,
	}

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xz import make_contourf_xz as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, y_level,
					interactive=False, plot_properties=plot_properties,
					plot_function_kwargs=plot_function_kwargs)

	#
	# Plot2d#2
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist_maccormack.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state2 = states[-1]

	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Horizontal velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 15],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 9.5,
		'cbar_orientation': 'vertical',
		'cbar_ax': (0, 1),
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, y_level,
					interactive=False, plot_properties=plot_properties,
					plot_function_kwargs=plot_function_kwargs)

	#
	# SubplotsAssembler
	#
	# Plot
	from tasmania.plot.assemblers import SubplotsAssembler
	assembler = SubplotsAssembler(1, 2, [monitor1, monitor2], interactive=False,
								  fontsize=12, figsize=[10, 5], tight_layout=False)
	assembler.store([state1, state2], save_dest=save_dest, show=False)


@image_comparison(baseline_images=['test_plot_2d_two_rows_two_columns'], extensions=['eps'])
def test_plot_2d_two_rows_two_columns():
	# Make sure the folder tests/baseline_images/test_subplots_assembler does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_subplots_assembler')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_2d_two_rows_two_columns.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Field to plot
	field_to_plot = 'x_velocity_at_u_locations'

	#
	# Plot2d#1
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state1 = states[-1]

	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_lim': [0, 500],
		'x_ticklabels': [],
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BuRd',
		'cbar_on': False,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 9.5,
	}

	# Instantiate the monitor
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xz import make_contourf_xz as plot_function
	monitor1 = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot2d#2
	#
	# Load data
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist_maccormack.pickle')
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state2 = states[-1]

	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_lim': [0, 500],
		'x_ticklabels': [],
		'y_lim': [0, 15],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 9.5,
		'cbar_orientation': 'vertical',
		'cbar_ax': (0, 1),
	}

	# Instantiate the monitor
	monitor2 = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
					plot_properties=plot_properties,
					plot_function_kwargs=plot_function_kwargs)

	#
	# Plot2d#3
	#
	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Pressure [hPa]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'field_factor': 1e-2,
		'cmap_name': 'Blues',
		'cbar_on': False,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
	}

	# Instantiate the monitor
	monitor3 = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
					plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)

	#
	# Plot2d#4
	#
	# Indices identifying the cross-section to visualize
	y_level = 0

	# Plot properties
	plot_properties = {
		'fontsize': 12,
		'title_left': 'Pressure [hPa]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 15],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'z_factor': 1e-3,
		'field_factor': 1e-2,
		'cmap_name': 'Blues',
		'cbar_on': True,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_orientation': 'vertical',
		'cbar_ax': (3, 4),
	}

	# Instantiate the monitor
	monitor4 = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
					plot_properties=plot_properties,
					plot_function_kwargs=plot_function_kwargs)

	#
	# SubplotsAssembler
	#
	# Plot
	from tasmania.plot.assemblers import SubplotsAssembler
	assembler = SubplotsAssembler(2, 2, [monitor1, monitor2, monitor3, monitor4],
								  interactive=False, fontsize=12, figsize=[8, 8],
								  tight_layout=False)
	assembler.store([state1, state2] * 2, save_dest=save_dest, show=False)


if __name__ == '__main__':
	pytest.main([__file__])
