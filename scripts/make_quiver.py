from loader import LoaderFactory
import tasmania as taz


#
# User inputs
#
filename = '../tests/baseline_datasets/isentropic_dry.nc'

x = None
y = None
z = -1

xcomp_name  = 'x_velocity'
xcomp_units = 'm s^-1'

ycomp_name  = 'y_velocity'
ycomp_units = 'm s^-1'

zcomp_name  = None
zcomp_units = None

scalar_name  = None
scalar_units = None

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
	'x_step': 2,
	'y_step': 2,
	'arrow_scale': None,
	'arrow_scale_units': None, # 'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'
	'arrow_headwidth': 5.0,
	'cmap_name': None,
	'cbar_on': False,
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
	'quiverkey_on': True,
	'quiverkey_loc': (0.83, 1.03),
	'quiverkey_length': 1.0,
	'quiverkey_label': '1 m s$^{-1}$',
	'quiverkey_label_loc': 'E',
	'quiverkey_fontproperties': {'size': 15},
	'draw_vertical_levels': False,
}


#
# Code
#
def get_drawer():
	loader = LoaderFactory.factory(filename)
	grid = loader.get_grid()

	drawer = taz.Quiver(
		grid, x=x, y=y, z=z,
		xcomp_name=xcomp_name, xcomp_units=xcomp_units,
		ycomp_name=ycomp_name, ycomp_units=ycomp_units,
		zcomp_name=zcomp_name, zcomp_units=zcomp_units,
		scalar_name=scalar_name, scalar_units=scalar_units,
		xaxis_name=xaxis_name, xaxis_units=xaxis_units,
		xaxis_y=xaxis_y, xaxis_z=xaxis_z,
		yaxis_name=yaxis_name, yaxis_units=yaxis_units,
		yaxis_x=yaxis_x, yaxis_z=yaxis_z,
		zaxis_name=zaxis_name, zaxis_units=zaxis_units,
		zaxis_x=zaxis_x, zaxis_y=zaxis_y,
		topography_units=topography_units,
		topography_x=topography_x, topography_y=topography_y,
		properties=drawer_properties
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
