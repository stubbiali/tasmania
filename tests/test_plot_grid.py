from matplotlib.testing.decorators import image_comparison
import os
import pickle
import pytest


@image_comparison(baseline_images=['test_plot_grid_xz'], extensions=['eps'])
def test_plot_grid_xz():
	# Dataset to load
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_dry.pickle')

	# Field to plot
	field_to_plot = 'grid'

	# Make sure the folder tests/baseline_images/test_plot_grid does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_plot_grid')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_grid_xz.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state  = states[-1]

	# Index identifying the cross-section to visualize
	y_level = int(grid.ny/2)

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'title_left': '$y = {}$ km'.format(grid.y.values[y_level]),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'hor_factor': 1e-3,
		'vert_factor': 1e-3,
		'linewidth': 1.2,
		'linecolor': 'gray',
		'fill_color': 'gray',
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.grid import plot_grid_xz as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, y_level, interactive=False,
				   fontsize=16, figsize=[7, 7], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


@image_comparison(baseline_images=['test_plot_grid_yz'], extensions=['eps'])
def test_plot_grid_yz():
	# Dataset to load
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_dry.pickle')

	# Field to plot
	field_to_plot = 'grid'

	# Make sure the folder tests/baseline_images/test_plot_grid does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_plot_grid')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_grid_yz.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Grab data from dataset
	with open(filename, 'rb') as data:
		grid   = pickle.load(data)
		states = pickle.load(data)
		state  = states[-1]

	# Index identifying the cross-section to visualize
	x_level = int(grid.nx/2)

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'title_left': '$x = {}$ km'.format(1e-3 * grid.x.values[x_level]),
		'x_label': '$y$ [km]',
		'x_lim': [-250, 250],
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'fontsize': 16,
		'hor_factor': 1e-3,
		'vert_factor': 1e-3,
		'linewidth': 1.2,
		'linecolor': 'black',
		'fill_color': 'brown',
	}

	# Plot
	from tasmania.plot.plot_monitors import Plot2d as Plot
	from tasmania.plot.grid import plot_grid_yz as plot_function
	monitor = Plot(grid, plot_function, field_to_plot, x_level, interactive=False,
				   fontsize=16, figsize=[7, 7], plot_properties=plot_properties,
				   plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
