from datetime import timedelta
import os
import pytest

from tasmania.plot.contourf import Contourf
from tasmania.plot.monitors import Plot, PlotComposite
from tasmania.plot.profile import LineProfile
from tasmania.plot.quiver import Quiver


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_composite')
def test_profile(isentropic_moist_sedimentation_data,
				 isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plot_composite_composer does exist
	baseline_dir = 'baseline_images/test_plot_composite'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	#
	# Plot#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Leapfrog',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}
	
	# Instantiate the left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (8, 7),
		'tight_layout': True,
	}

	# Instantiate the monitor
	monitor = PlotComposite(1, 2, (plot1, plot2), False, figure_properties)

	# Plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_composite')
def test_profile_share_yaxis(isentropic_moist_sedimentation_data,
							 isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plot_composite_composer does exist
	baseline_dir = 'baseline_images/test_plot_composite'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_share_yaxis_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	#
	# Plot#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Leapfrog',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 2.0],
		'y_ticklabels': (),
		'grid_on': True,
	}

	# Instantiate the right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (8, 7),
		'tight_layout': True,
	}

	# Instantiate the monitor
	monitor = PlotComposite(1, 2, (plot1, plot2), False, figure_properties)

	# Plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_composite')
def test_profile_share_xaxis(isentropic_moist_sedimentation_data,
							 isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plot_composite_composer does exist
	baseline_dir = 'baseline_images/test_plot_composite'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_profile_share_xaxis_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'accumulated_precipitation'
	field_units = 'mm'

	#
	# Plot#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'Leapfrog',
		'x_lim': [0, 500],
		'x_ticklabels': (),
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 3.0],
		'grid_on': True,
	}

	# Instantiate the left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Indices identifying the cross-line to visualize
	y, z = 0, -1

	# Drawer properties
	drawer_properties = {
		'linecolor': 'red',
		'linestyle': '--',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = LineProfile(grid, field_name, field_units, y=y, z=z, axis_units='km',
						 **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 16,
		'title_left': 'MacCormack',
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': 'Accumulated precipitation [mm]',
		'y_lim': [0, 2.0],
		'grid_on': True,
	}

	# Instantiate the right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (6, 9),
		'tight_layout': True,
	}

	# Instantiate the monitor
	monitor = PlotComposite(2, 1, (plot1, plot2), False, figure_properties)

	# Plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_composite')
def test_plot2d_r1c2(isentropic_moist_sedimentation_data,
								    isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plot_composite does exist
	baseline_dir = 'baseline_images/test_plot_composite'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_plot2d_r1c2_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	# Field to plot
	field_name  = 'x_velocity_at_u_locations'
	field_units = None

	#
	# Plot#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'cmap_name': 'BuRd',
		'cbar_on': False,
		'cbar_levels': 22,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 10.5,
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(grid, field_name, field_units, y=y, xaxis_units='km',
					  zaxis_name='height', zaxis_units='km', **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$z$ [km]',
		'y_lim': [0, 14],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Instantiate the left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'fontsize': 14,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 22,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 10.5,
		'cbar_ax': (0, 1),
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(grid, field_name, field_units, y=y, xaxis_units='km',
					  zaxis_name='height', zaxis_units='km', **drawer_properties)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 14],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Instantiate the right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 14,
		'figsize': (9, 6),
		'tight_layout': False,
	}

	# Instantiate the monitor
	monitor = PlotComposite(1, 2, (plot1, plot2), False, figure_properties)

	# Plot
	monitor.store((state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


@pytest.mark.mpl_image_compare(baseline_dir='baseline_images/test_plot_composite')
def test_plot2d_r2c2(isentropic_moist_sedimentation_data,
					 isentropic_moist_sedimentation_evaporation_data):
	# Make sure the folder tests/baseline_images/test_plot_composite does exist
	baseline_dir = 'baseline_images/test_plot_composite'
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	save_dest = os.path.join(baseline_dir, 'test_plot2d_r2c2_nompl.eps')
	if os.path.exists(save_dest):
		os.remove(save_dest)

	#
	# Plot#1
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_evaporation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state1 = states[-1]

	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'cmap_name': 'BuRd',
		'cbar_on': False,
		'cbar_levels': 22,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 10.5,
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(
		grid, 'x_velocity_at_u_locations', 'm s^-1', y=y,
		xaxis_units='km', zaxis_name='height', zaxis_units='km',
		**drawer_properties,
	)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_lim': [0, 500],
		'x_ticklabels': [],
		'y_label': '$z$ [km]',
		'y_lim': [0, 14],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Instantiate the upper-left collaborator
	plot1 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#2
	#
	# Load data
	grid, states = isentropic_moist_sedimentation_data
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state2 = states[-1]

	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'fontsize': 14,
		'cmap_name': 'BuRd',
		'cbar_on': True,
		'cbar_levels': 22,
		'cbar_ticks_step': 2,
		'cbar_center': 15,
		'cbar_half_width': 10.5,
		'cbar_ax': (0, 1),
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(
		grid, 'x_velocity_at_u_locations', 'm s^-1', y=y,
		xaxis_units='km', zaxis_name='height', zaxis_units='km',
		**drawer_properties,
	)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': '$x$-velocity [m s$^{-1}$]',
		'title_right': str(timedelta(hours=6)),
		'x_lim': [0, 500],
		'x_ticklabels': [],
		'y_lim': [0, 14],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Instantiate the upper-right collaborator
	plot2 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#3
	#
	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'cmap_name': 'Blues',
		'cbar_on': False,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(
		grid, 'air_pressure_on_interface_levels', 'hPa', y=y,
		xaxis_units='km', zaxis_name='height', zaxis_units='km',
		**drawer_properties,
	)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': 'Pressure [hPa]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$z$ [km]',
		'y_lim': [0, 14],
		'text': 'LF',
		'text_loc': 'upper right',
	}

	# Instantiate the lower-left collaborator
	plot3 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# Plot#4
	#
	# Index identifying the cross-section to visualize
	y = 0

	# Drawer properties
	drawer_properties = {
		'cmap_name': 'Blues',
		'cbar_on': True,
		'cbar_levels': 20,
		'cbar_ticks_step': 2,
		'cbar_ax': (3, 4),
		'linecolor': 'black',
		'linewidth': 1.5,
	}

	# Instantiate the drawer
	drawer = Contourf(
		grid, 'air_pressure_on_interface_levels', 'hPa', y=y,
		xaxis_units='km', zaxis_name='height', zaxis_units='km',
		**drawer_properties,
	)

	# Axes properties
	axes_properties = {
		'fontsize': 14,
		'title_left': 'Pressure [hPa]',
		'title_right': str(timedelta(hours=6)),
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_lim': [0, 14],
		'y_ticklabels': [],
		'text': 'MC',
		'text_loc': 'upper right',
	}

	# Instantiate the lower-left collaborator
	plot4 = Plot(drawer, False, axes_properties=axes_properties)

	#
	# PlotComposite
	#
	# Figure properties
	figure_properties = {
		'fontsize': 14,
		'figsize': (9, 12),
		'tight_layout': False,
	}

	# Instantiate the monitor
	monitor = PlotComposite(2, 2, (plot1, plot2, plot3, plot4), False, figure_properties)

	# Plot
	monitor.store((state1, state2, state1, state2), save_dest=save_dest)

	assert os.path.exists(save_dest)

	return monitor.figure


if __name__ == '__main__':
	pytest.main([__file__])
