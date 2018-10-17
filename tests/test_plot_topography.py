from matplotlib.testing.decorators import image_comparison
import os
import pytest


#@image_comparison(baseline_images=['test_plot_topography_3d'], extensions=['eps'], tol=0.15)
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_topography')
def test_plot_topography_3d(isentropic_dry_data):
	# Make sure the folder tests/baseline_images/test_plot_topography does exist
	baseline_dir = 'baseline_images/test_plot_topography'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_plot_topography_3d.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Load and grab data
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$y$ [km]',
		'y_lim': [-250, 250],
		'z_label': '$z$ [km]',
		'z_lim': None,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BrBG_r',
		'cbar_on': True,
		'cbar_orientation': 'vertical',
	}

	from tasmania.plot.topography import plot_topography_3d as plot_function
	from tasmania.plot.plot_monitors import Plot3d
	monitor = Plot3d(grid, plot_function, 'topography', interactive=False,
					 fontsize=16, figsize=[8, 7],
					 plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
