from loader import LoaderFactory
import tasmania as taz


#
# User inputs
#
plot_function_kwargs = {
	'fontsize': 16,
	'x_factor': 1e-3,
	'z_factor': 1e-3,
	'field_bias': 10., #10.,
	'field_factor': 1e4, #1e4,
	'draw_grid': False,
	'cmap_name': 'BuRd', #'BuRd',
	'cbar_on': True,
	'cbar_levels': 18,
	'cbar_ticks_step': 4,
	'cbar_ticks_pos': 'center',
	'cbar_center': 0,
	'cbar_half_width': 470, #8.5, #470, #220,
	'cbar_x_label': '',
	'cbar_y_label': '',
	'cbar_title': '',
	'cbar_orientation': 'horizontal',
	'cbar_ax': None,
}

plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '$u\' = u - \\bar{u}$  [10$^{-4}$ m s$^{-1}$]',
	'title_right': '',
	'x_label': '$x$ [km]',
	'x_lim': [-120, 120],
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': None,
	'x_ticklabels': None,
	'xaxis_minor_ticks_visible': True,
	'xaxis_visible': True,
	'y_label': '$z$ [km]',
	'y_lim': [0, 12],
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None,
	'y_ticklabels': None,
	'yaxis_minor_ticks_visible': True,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	'legend_on': False,
	'legend_loc': 'best',
	'text': None,
	'text_loc': 'upper right',
	'grid_on': False,
	'grid_properties': {'linestyle': ':'},
}

filename = '../data/isentropic_convergence_rk3cosmo_fifth_order_upwind_nx801_dt1_nt96000.nc'
time_level = -1
field_to_plot = 'x_velocity_at_u_locations'
level = 0
fontsize = 16
figsize = (7, 8)
tight_layout = True
print_time = 'elapsed' # 'elapsed', 'absolute'


#
# Code
#
def get_artist(tlevel=None):
	tlevel = tlevel if tlevel is not None else time_level

	# Load the data
	grid, states = taz.load_netcdf_dataset(filename)
	grid.update_topography(states[tlevel]['time'] - states[0]['time'])
	state = states[tlevel]

	# Print time
	if print_time == 'elapsed':
		plot_properties['title_right'] = str(state['time'] - states[0]['time'])
	elif print_time == 'absolute':
		plot_properties['title_right'] = str(state['time'])

	# Instantiate the artist
	artist = taz.Plot2d(grid, taz.make_contourf_xz, field_to_plot, level,
						interactive=False, fontsize=16, figsize=figsize,
						tight_layout=tight_layout, plot_properties=plot_properties,
				   		plot_function_kwargs=plot_function_kwargs)

	return artist, state


def get_loader():
	return LoaderFactory(filename)


if __name__ == '__main__':
	artist, state = get_artist()
	artist.store(state, show=True)
