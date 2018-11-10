from datetime import datetime
from loader import LoaderFactory
import tasmania as taz


#
# User inputs
#
filename = '../tests/baseline_datasets/isentropic_dry.nc'

field_name  = 'x_velocity'
field_units = 'm s^-1'

x = 25
y = 25
z = None

axis_name  = 'height_on_interface_levels'
axis_units = 'km'
axis_x = None
axis_y = None
axis_z = None

time_mode  = 'elapsed'
init_time  = None
time_units = 'hr'

drawer_properties = {
	'fontsize': 16,
	'cmap_name': 'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': 15,
	'cbar_half_width': None, #8.5, #470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
	'draw_vertical_levels': False,
	'linecolor': 'black',
	'linewidth': 1.2,
}


#
# Code
#
def get_drawer():
	loader = LoaderFactory.factory(filename)
	grid = loader.get_grid()

	drawer = taz.HovmollerDiagram(
		grid, field_name, field_units, x=x, y=y, z=z,
		axis_name=axis_name, axis_units=axis_units,
		axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
		time_mode=time_mode, init_time=init_time,
		time_units=time_units, properties=drawer_properties
	)

	return drawer


def get_state(tlevel, drawer, axes_properties=None, print_time=None):
	loader = LoaderFactory.factory(filename)

	tlevel = loader.nt + tlevel if tlevel < 0 else tlevel

	for k in range(0, tlevel-1):
		drawer(loader.get_state(k))

	return loader.get_state(tlevel)
