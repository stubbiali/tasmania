import os
import pytest

from tasmania.plot.trackers import HovmollerDiagram
from tasmania.plot.monitors import Plot


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_hovmoller')
def test_x(isentropic_dry_data):
	# Field to plot
	field_name  = 'horizontal_velocity'
	field_units = 'km hr^-1'

	# Make sure the folder tests/baseline_images/test_hovmoller does exist
	baseline_dir = 'baseline_images/test_hovmoller'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_x_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data

	# Indices identifying the line to visualize
	y, z = 25, -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 18,
		'cbar_ticks_step': 4,
		'cbar_ticks_pos': 'center',
		'cbar_center': 15,
		'cbar_x_label': 'Surface velocity [km h$^{-1}$]',
		'cbar_orientation': 'horizontal',
	}

	# Instantiate the drawer
	drawer = HovmollerDiagram(grid, field_name, field_units, y=y, z=z, axis_units='km',
							  time_mode='elapsed', time_units='hr', **drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$y = $0 km',
		'x_label': '$x$ [km]',
		'y_label': 'Elapsed time [h]',
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	for state in states[:-1]:
		drawer(state)
	monitor.store(states[-1], save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_hovmoller')
def test_z(isentropic_dry_data):
	# Field to plot
	field_name  = 'x_velocity_at_u_locations'
	field_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_hovmoller does exist
	baseline_dir = 'baseline_images/test_hovmoller'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_z_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data

	# Indices identifying the line to visualize
	x, y = 25, 25

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 18,
		'cbar_ticks_step': 4,
		'cbar_ticks_pos': 'center',
		'cbar_center': 15,
		'cbar_orientation': 'horizontal',
	}

	# Instantiate the drawer
	drawer = HovmollerDiagram(grid, field_name, field_units, x=x, y=y,
							  time_mode='elapsed', **drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x$-velocity [m s$^{-1}$] at ($x = $0 km, $y = $0 km)',
		'x_label': 'Elapsed time [s]',
		'y_label': '$\\theta$ [K]',
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	for state in states[:-1]:
		drawer(state)
	monitor.store(states[-1], save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_lineprofile(isentropic_dry_data())
