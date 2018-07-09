from matplotlib.testing.decorators import image_comparison
import os
import pickle
import pytest


@image_comparison(baseline_images=['test_quiver_xy_velocity'], extensions=['eps'], tol=1.5e-1)
def test_quiver_xy_velocity():
	# Dataset to load
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_dry.pickle')

	# Field to plot
	field_to_plot = 'horizontal_velocity'

	# Make sure the folder tests/baseline_images/test_quiver_xy does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_quiver_xy')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_quiver_xy_velocity.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state  = states[-1]

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
		'x_step': 2,
		'y_factor': 1e-3,
		'y_step': 2,
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
	from tasmania.plot.quiver_xy import make_quiver_xy as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, z_level, interactive=True,
				   fontsize=16, figsize=[7, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_quiver_xy_velocity_bw'], extensions=['eps'], tol=1.5e-1)
def test_quiver_xy_velocity_bw():
	# Dataset to load
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_dry.pickle')

	# Field to plot
	field_to_plot = 'horizontal_velocity'

	# Make sure the folder tests/baseline_images/test_quiver_xy does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_quiver_xy')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_quiver_xy_velocity_bw.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state  = states[-1]

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
		'x_step': 2,
		'y_factor': 1e-3,
		'y_step': 2,
		'field_bias': 0.,
		'field_factor': 1.,
		'cmap_name': None,
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.quiver_xy import make_quiver_xy as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, z_level, interactive=True,
				   fontsize=16, figsize=[8, 8], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
