import os
import pytest

from tasmania.plot.contour import Contour
from tasmania.plot.monitors import Plot


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_contour')
def test_contour_xy_velocity(isentropic_dry_data):
	# Field to plot
	field_name  = 'horizontal_velocity'
	field_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_contour does exist
	baseline_dir = 'baseline_images/test_contour'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_contour_xy_velocity_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'alpha': 1.0,
		'colors': 'black',
	}

	# Instantiate the drawer
	drawer = Contour(grid, field_name, field_units, z=z,
					 xaxis_units='km', yaxis_units='km',
					 properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Surface horizontal velocity [m s$^{-1}$]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_contour')
def test_contour_xy_pressure(isentropic_dry_data):
	# Field to plot
	field_name  = 'air_pressure_on_interface_levels'
	field_units = 'hPa'

	# Make sure the folder tests/baseline_images/test_contour does exist
	baseline_dir = 'baseline_images/test_contour'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_contour_xy_pressure_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grad data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'alpha': 0.1,
	}

	# Instantiate the drawer
	drawer = Contour(grid, field_name, field_units, z=z,
					 xaxis_units='km', yaxis_units='km',
					 properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Surface pressure [hPa]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$y$ [km]',
		'y_lim': None,
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_contour')
def test_contour_xz_velocity(isentropic_dry_data):
	# Field to plot
	field_name  = 'x_velocity_at_u_locations'
	field_units = 'm s^-1'

	# Make sure the folder tests/baseline_images/test_contour does exist
	baseline_dir = 'baseline_images/test_contour'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_contour_xz_velocity_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	y = int(grid.ny/2)

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'draw_vertical_levels': True,
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contour(grid, field_name, field_units, y=y,
					 xaxis_units='km', zaxis_name='height', zaxis_units='km',
					 properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': None,
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_contour')
def test_contour_yz_velocity(isentropic_dry_data):
	# Field to plot
	field_name  = 'y_velocity_at_v_locations'
	field_units = 'km hr^-1'

	# Make sure the folder tests/baseline_images/test_contour does exist
	baseline_dir = 'baseline_images/test_contour'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_contour_yz_velocity_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Grab data from dataset
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time']-states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section to visualize
	x = int(grid.nx/2)

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'draw_vertical_levels': True,
		'linecolor': 'black',
		'linewidth': 1.3,
	}

	# Instantiate the drawer
	drawer = Contour(grid, field_name, field_units, x=x, yaxis_units='km',
					 zaxis_name='height_on_interface_levels', zaxis_units='km',
					 properties=drawer_properties)

	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$y$-velocity [km h$^{-1}$]',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$y$ [km]',
		'x_lim': None,
		'y_label': '$z$ [km]',
		'y_lim': [0, 15],
	}

	# Instantiate the monitor
	monitor = Plot(drawer, False, figure_properties, axes_properties)

	# Plot
	monitor.store(state, save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_contour_xz_velocity(isentropic_dry_data())
