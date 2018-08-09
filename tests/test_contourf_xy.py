from matplotlib.testing.decorators import image_comparison
import os
import pytest
import sys
sys.path.append(os.path.dirname(__file__))

from conftest import isentropic_dry_data


@image_comparison(baseline_images=['test_contourf_xy_velocity'], extensions=['eps'])
def test_contourf_xy_velocity():
	# Field to plot
	field_to_plot = 'horizontal_velocity'

	# Make sure the folder tests/baseline_images/test_contourf_xy does exist
	baseline_dir = 'baseline_images/test_contourf_xy'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_contourf_xy_velocity.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z_level = -1

	# Plot properties
	plot_properties = {
		'title_left': 'Horizontal velocity [m s$^{-1}$]',
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
		'text': 'MC',
		'text_loc': 'upper right',
	}

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

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xy import make_contourf_xy as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, z_level, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_contourf_xy_pressure'], extensions=['eps'])
def test_contourf_xy_pressure():
	# Field to plot
	field_to_plot = 'air_pressure_on_interface_levels'

	# Make sure the folder tests/baseline_images/test_contourf_xy does exist
	baseline_dir = 'baseline_images/test_contourf_xy'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_contourf_xy_pressure.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grad data from dataset
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z_level = -1

	# Plot properties
	plot_properties = {
		'title_left': 'Pressure [hPa]',
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'x_factor': 1e-3,
		'y_factor': 1e-3,
		'field_bias': 0.,
		'field_factor': 1.e-2,
		'cmap_name': 'Blues',
		'cbar_levels': 11,
		'cbar_ticks_step': 2,
		'cbar_ticks_pos': 'interface',
		'cbar_center': None,
		'cbar_half_width': None,
		'cbar_x_label': '',
		'cbar_y_label': '',
		'cbar_title': '',
		'cbar_orientation': 'horizontal',
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.contourf_xy import make_contourf_xy as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, z_level, interactive=False,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
