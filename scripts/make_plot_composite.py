from loader import LoaderComposite
import tasmania as taz


#
# User inputs
#
nrows = 1
ncols = 2

modules = [
	'make_plot',
	'make_plot_1',
]

tlevel = -1

figure_properties = {
	'fontsize': 16,
	'figsize': (9, 6),
	'tight_layout': True
}

save_dest = None


#
# Code
#
def get_plot():
	subplots = []

	for module in modules:
		import_str = 'from {} import get_plot as get_subplot'.format(module)
		exec(import_str)
		subplots.append(locals()['get_subplot']())

	plot = taz.PlotComposite(nrows, ncols, subplots, interactive=False,
							 figure_properties=figure_properties)

	return plot


def get_states(tlevel, plot):
	subplots = plot.artists
	states = []

	for module, subplot in zip(modules, subplots):
		import_str = 'from {} import get_states'.format(module)
		exec(import_str)
		states.append(locals()['get_states'](tlevel, subplot))

	return states


if __name__ == '__main__':
	plot = get_plot()
	states = get_states(tlevel, plot)
	plot.store(states, save_dest=save_dest, show=True)

