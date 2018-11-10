import os
import pytest

from tasmania.plot.contourf import Contourf
from tasmania.plot.monitors import Plot
from tasmania.plot.profile import LineProfile
from tasmania.plot.quiver import Quiver


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plots')
def test_profile_x(isentropic_moist_sedimentation_data,
				   isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plots does exist
	baseline_dir = 'baseline_images/test_plots'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_x_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	#
	# Drawer#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
		'legend_label': 'LF, evap. ON',
	}
	
	# Instantiate the drawer
	drawer1 = LineProfile(grid, field_name, field_units, y=y, z=z,
						  axis_units='km', properties=drawer_properties)

	#
	# Drawer#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'linecolor': 'green',
		'linestyle': '--',
		'linewidth': 1.5,
		'legend_label': 'MC, evap. OFF',
	}

	# Instantiate the drawer
	drawer2 = LineProfile(grid, field_name, field_units, y=y, z=z,
						  axis_units='km', properties=drawer_properties)

	#
	# Plot
	#
	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$y = ${} km, $\\theta = ${} K'.format(
			grid.y.values[0]/1e3, grid.z.values[-1]),
		'title_right': str(state1['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 1.0],
		'legend_on': True,
		'legend_loc': 'best',
		'legend_framealpha': 1.0,
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot((drawer1, drawer2), False, figure_properties, axes_properties)

	# plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plots')
def test_profile_z(isentropic_moist_sedimentation_data,
				   isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plots does exist
	baseline_dir = 'baseline_images/test_plots'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_z_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'mass_fraction_of_cloud_liquid_water_in_air'
	field_units = 'g kg^-1'

	#
	# Drawer#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	x, y = 40, 0

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
		'legend_label': 'LF, evap. ON',
	}

	# Instantiate the monitor
	drawer1 = LineProfile(grid, field_name, field_units, x=x, y=y,
						  properties=drawer_properties)

	#
	# Drawer#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-section to visualize
	x, y = 40, 0

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'linecolor': 'lightblue',
		'linestyle': '--',
		'linewidth': 1.5,
		'legend_label': 'MC, evap. OFF',
	}

	# Instantiate the monitor
	drawer2 = LineProfile(grid, field_name, field_units, x=x, y=y,
						  properties=drawer_properties)

	#
	# Plot
	#
	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x = ${} km, $y = ${} km'.format(
			grid.x.values[40]/1e3, grid.y.values[0]/1e3),
		'title_right': str(state1['time'] - states[0]['time']),
		'x_label': 'Cloud liquid water [g kg$^{-1}$]',
		'x_lim': [0, 0.12],
		'y_label': '$\\theta$ [K]',
		'legend_on': True,
		'legend_loc': 'best',
		'grid_on': True,
	}

	# Instantiate the monitor
	monitor = Plot((drawer1, drawer2), False, figure_properties, axes_properties)

	# plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plots')
def test_plot2d(isentropic_dry_data):
	# Make sure the folder tests/baseline_images/test_plots does exist
	baseline_dir = 'baseline_images/test_plots'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_plot2d_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	#
	# Drawer#1
	#
	# Load data
	grid, states = isentropic_dry_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Index identifying the cross-section
	z = -1

	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
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

	# Instantiate the drawer
	drawer1 = Contourf(grid, 'horizontal_velocity', 'm s^-1', z=z,
					   xaxis_units='km', yaxis_units='km', properties=drawer_properties)

	#
	# Drawer#2
	#
	# Drawer properties
	drawer_properties = {
		'fontsize': 16,
		'x_step': 2,
		'y_step': 2,
		'cmap_name': None,
		'alpha': 0.5,
	}

	# Instantiate the monitor
	drawer2 = Quiver(grid, z=z,
					 xcomp_name='x_velocity', ycomp_name='y_velocity',
					 xaxis_units='km', yaxis_units='km', properties=drawer_properties)

	#
	# Plot
	#
	# Figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 8),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Horizontal velocity [m s$^{-1}$] at the surface',
		'title_right': str(state['time'] - states[0]['time']),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$y$ [km]',
		'y_lim': [-250, 250],
	}

	# Instantiate the monitor
	monitor = Plot((drawer1, drawer2), False, figure_properties, axes_properties)

	# plot
	monitor.store((state, state), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
