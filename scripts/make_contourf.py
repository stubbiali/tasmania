from datetime import datetime
from loader import LoaderFactory
import tasmania as taz


#
# User inputs
#
filename = '../tests/baseline_datasets/isentropic_dry.nc'

field_name  = 'x_velocity'
field_units = 'km hr^-1'

x = None
y = 25
z = None

xaxis_name  = 'x'
xaxis_units = 'km'
xaxis_y = None
xaxis_z = None

yaxis_name  = 'y'
yaxis_units = 'km'
yaxis_x = None
yaxis_z = None

zaxis_name  = 'height'
zaxis_units = 'km'
zaxis_x = None
zaxis_y = None

topography_units = 'km'
topography_x = None
topography_y = None

drawer_properties = {
	'fontsize': 16,
	'cmap_name': 'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': 15 * 3.6,
	'cbar_half_width': None, #8.5, #470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
	'draw_vertical_levels': True,
	'linecolor': 'black',
	'linewidth': 1.2,
}


#
# Code
#
def get_drawer():
	loader = LoaderFactory.factory(filename)
	grid = loader.get_grid()

	drawer = taz.Contourf(
		grid, field_name, field_units, x=x, y=y, z=z,
		xaxis_name=xaxis_name, xaxis_units=xaxis_units,
		xaxis_y=xaxis_y, xaxis_z=xaxis_z,
		yaxis_name=yaxis_name, yaxis_units=yaxis_units,
		yaxis_x=yaxis_x, yaxis_z=yaxis_z,
		zaxis_name=zaxis_name, zaxis_units=zaxis_units,
		zaxis_x=zaxis_x, zaxis_y=zaxis_y,
		topography_units=topography_units,
		topography_x=topography_x, topography_y=topography_y,
		**drawer_properties
	)

	return drawer


def get_state(tlevel, drawer=None, axes_properties=None, print_time=None):
	loader = LoaderFactory.factory(filename)
	state = loader.get_state(tlevel)

	if axes_properties is not None:
		if print_time == 'elapsed':
			init_time = loader.get_state(0)['time']
			axes_properties['title_right'] = str(state['time'] - init_time)
		elif print_time == 'absolute':
			axes_properties['title_right'] = str(state['time'])

	return state
