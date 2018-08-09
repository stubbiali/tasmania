from matplotlib.testing.decorators import image_comparison
import os
import pytest
import sys
sys.path.append(os.path.dirname(__file__))

from conftest import isentropic_dry_data, isentropic_moist_sedimentation_data


@image_comparison(baseline_images=['test_profile_1d_x'], extensions=['eps'])
def test_profile_1d_x():
	# Field to plot
	field_to_plot = 'accumulated_precipitation'

	# Make sure the folder tests/baseline_images/test_profile_1d does exist
	baseline_dir = 'baseline_images/test_profile_1d'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_profile_1d_x.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {1: 0, 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': '$y = ${} km, $\\theta = ${} K'.format(grid.y[0]/1e3, grid.z[-1]),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 0.9],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}
	
	# Plot
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_profile_1d_y'], extensions=['eps'])
def test_profile_1d_y():
	# Field to plot
	field_to_plot = 'y_velocity_at_v_locations'

	# Make sure the folder tests/baseline_images/test_profile_1d does exist
	baseline_dir = 'baseline_images/test_profile_1d'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_profile_1d_y.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {0: int(grid.nx/2), 2: -1}

	# Plot properties
	plot_properties = {
		'title_left': '$x = ${} km, $\\theta = ${} K'.format(grid.x[int(grid.nx/2)]/1e3, grid.z[-1]),
		'x_label': '$y$ [km]',
		'x_lim': None,
		'y_label': '$y$-velocity [m s$^{-1}$]',
		'y_lim': [-4.5, 4.5],
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1.,
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_horizontal_profile as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_profile_1d_z'], extensions=['eps'])
def test_profile_1d_z():
	# Field to plot
	field_to_plot = 'mass_fraction_of_cloud_liquid_water_in_air'

	# Make sure the folder tests/baseline_images/test_profile_1d does exist
	baseline_dir = 'baseline_images/test_profile_1d'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_profile_1d_z.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {0: 40, 1: 0}

	# Plot properties
	plot_properties = {
		'title_left': '$x = ${} km, $y = ${} km'.format(grid.x[40]/1e3, grid.y[0]/1e3),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.12],
		'y_label': '$\\theta$ [K]',
		'y_lim': None,
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e3,
		'y_factor': 1.,
		'linecolor': 'lightblue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import plot_vertical_profile as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_profile_1d_height'], extensions=['eps'])
def test_profile_1d_height():
	# Field to plot
	field_to_plot = 'mass_fraction_of_cloud_liquid_water_in_air'

	# Make sure the folder tests/baseline_images/test_profile_1d does exist
	baseline_dir = 'baseline_images/test_profile_1d'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_profile_1d_height.eps')
	save_dest = None if os.path.exists(baseline_img + '.eps') else baseline_img

	# Grab data from dataset
	grid, states = isentropic_moist_sedimentation_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Indices identifying the cross-line to visualize
	levels = {0: 40, 1: 0}

	# Plot properties
	plot_properties = {
		'title_left': '$x = ${} km, $y = ${} km'.format(grid.x[40]/1e3, grid.y[0]/1e3),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.12],
		'y_label': '$z$ [km]',
		'y_lim': None,
		'grid_on': True,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e3,
		'y_factor': 1e-3,
		'linecolor': 'lightblue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot1d as Plot
	from tasmania.plot.profile_1d import \
		plot_vertical_profile_with_respect_to_vertical_height as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, levels, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
