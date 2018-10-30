from loader import LoaderComposite
import tasmania as taz


#
# User inputs
#
modules = [
	'make_contourf_xy',
	'make_quiver_xy',
]

fontsize = 16

plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': 'Surface velocity [m s$^{-1}$]',
	'title_right': '',
	'x_label': '$x$ [km]',
	'x_lim': [-200, 200],
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': None,
	'x_ticklabels': None,
	'xaxis_minor_ticks_visible': True,
	'xaxis_visible': True,
	'y_label': '$y$ [km]',
	'y_lim': [-200, 200],
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None,
	'y_ticklabels': None,
	'yaxis_minor_ticks_visible': True,
	'yaxis_visible': False,
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
	'text_loc': '',
	'grid_on': False,
	'grid_properties': {'linestyle': ':'},
}

figsize = (7, 8)
tight_layout = True


#
# Code
#
def get_artist(tlevel=None):
	slaves = []
	states = []

	for module in modules:
		import_str = 'from {} import get_artist as get_slave'.format(module)
		exec(import_str)
		slave, state = locals()['get_slave'](tlevel)
		slaves.append(slave)
		states.append(state)

	artist = taz.PlotsOverlapper(slaves, interactive=False, fontsize=fontsize,
								 figsize=figsize, tight_layout=tight_layout,
							     plot_properties=plot_properties)

	return artist, states


def get_loader():
	subloaders = []

	for module in modules:
		import_str = 'from {} import get_loader'.format(module)
		exec(import_str)
		subloaders.append(locals()['get_loader']())

	return LoaderComposite(subloaders)


if __name__ == '__main__':
	artist, states = get_artist()
	artist.store(states, show=True)
