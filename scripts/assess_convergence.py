from matplotlib import rcParams, ticker
import matplotlib.pyplot as plt
import numpy as np
import tasmania as taz


#
# User inputs
#
field_name  = 'x_velocity'
field_units = 'm s^-1'

reference_dataset = {
	'filename':
		'../data/isentropic_convergence_rk3cosmo_fifth_order_upwind_nx801_dt1_nt96000.nc',
	'xslice': slice(320, 481), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
	#'xslice': slice(0, 401), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
}

datasets = [
	[
		{
			'filename':
				'../data/isentropic_convergence_rk2_third_order_upwind_nx51_dt20_nt6000_bis.nc',
			'xslice': slice(20, 31), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 16, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk2_third_order_upwind_nx101_dt10_nt12000_bis.nc',
			'xslice': slice(40, 61), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 8, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk2_third_order_upwind_nx201_dt5_nt24000_bis.nc',
			'xslice': slice(80, 121), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 4, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk2_third_order_upwind_nx401_dt2_nt48000_bis.nc',
			'xslice': slice(160, 241), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 2, 'ysampling': 1, 'zsampling': 1,
		},
	],
	[
		{
			'filename':
				'../data/isentropic_convergence_rk3_fifth_order_upwind_nx51_dt20_nt6000.nc',
			'xslice': slice(20, 31), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 16, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk3_fifth_order_upwind_nx101_dt10_nt12000.nc',
			'xslice': slice(40, 61), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 8, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk3_fifth_order_upwind_nx201_dt5_nt24000.nc',
			'xslice': slice(80, 121), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 4, 'ysampling': 1, 'zsampling': 1,
		},
		{
			'filename':
				'../data/isentropic_convergence_rk3_fifth_order_upwind_nx401_dt2_nt48000.nc',
			'xslice': slice(160, 241), 'yslice': slice(0, 1), 'zslice': slice(200, 300),
			'xsampling': 2, 'ysampling': 1, 'zsampling': 1,
		},
	],
]

figsize = (7, 7)
fontsize = 16

xs = [
	[np.log(x) for x in [20, 10, 5, 2.5]],
	[np.log(x) for x in [20, 10, 5, 2.5]],
]

refs_x = [
	[np.log(x) for x in [10, 5, 2.5]],
	#[np.log(x) for x in [10, 5, 2.5]],
	[np.log(x) for x in [10, 5, 2.5]],
]
refs_y = [
	[4e-4 * (x/2.5)**2 for x in [10, 5, 2.5]],
	#[3e-3 * (x/10)**2.5 for x in [10, 5, 2.5]],
	[3e-3 * (x/10)**3 for x in [10, 5, 2.5]],
]

linestyles = [
	'-',
	'-',
]

linecolors = [
	'blue',
	'red',
]

linewidths = [
	1.5,
	1.5,
]

markers = [
	'o',
	's',
]

markersizes = [
	6.5,
	6.5,
]

markerfacecolors = [
	'white',
	'white',
]

labels = [
	'RK2',
	'RK3',
]

plot_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '',
	'title_right': '',
	'x_label': '$\\Delta x$ [km]',
	'x_lim': [np.log(30), np.log(1.5)],
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': [np.log(2.5), np.log(5), np.log(10), np.log(20)],
	'x_ticklabels': [0.55, 1.1, 2.2, 4.4],
	'xaxis_minor_ticks_visible': False,
	'xaxis_visible': True,
	'y_label': 'TRER [m s$^{-1}$]',
	'y_lim': (4e-5, 1e-2),
	'invert_yaxis': False,
	'y_scale': 'log',
	'y_ticks': [1e-4, 1e-3, 1e-2],
	'y_ticklabels': ['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
	'yaxis_minor_ticks_visible': False,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	'legend_on': True,
	'legend_loc': 'best',
	'legend_framealpha': 1.0,
	'text': None,
	'text_loc': '',
	'grid_on': True,
	'grid_properties': {'linestyle': ':'},
}


#
# Code
#
def _get_raw_field(state, name, units):
	if name in state.keys():
		return state[name].to_units(units).values
	elif name == 'x_velocity':
		try:
			s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1')
			su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1')
			u  = su / s
			u.attrs['units'] = 'm s^-1'

			return u.to_units(units).values
		except KeyError:
			pass


if __name__ == '__main__':
	# Get reference solution
	print(reference_dataset['filename'])
	grid_r, states = taz.load_netcdf_dataset(reference_dataset['filename'])
	state_r = states[-1]
	raw_field_r = _get_raw_field(state_r, field_name, field_units)
	refsol = raw_field_r[reference_dataset['xslice'],
						 reference_dataset['yslice'],
						 reference_dataset['zslice']]

	# Scan all datasets
	err = []
	for i in range(len(datasets)):
		err.append([])
		for ds in datasets[i]:
			print(ds['filename'])

			grid, states = taz.load_netcdf_dataset(ds['filename'])
			state = states[-1]
			raw_field = _get_raw_field(state, field_name, field_units)
			sol = raw_field[ds['xslice'], ds['yslice'], ds['zslice']]
			rsol = refsol[::ds['xsampling'], ::ds['ysampling'], ::ds['zsampling']]

			err[i].append((np.sum(np.abs(sol - rsol)**2) / (sol.shape[0]*sol.shape[1]*sol.shape[2]))**0.5)

	# Open a window
	fig, ax = taz.get_figure_and_axes(figsize=figsize, fontsize=fontsize)

	# Plot
	for i in range(len(datasets)):
		ax.plot(xs[i], err[i], linestyle=linestyles[i], color=linecolors[i], linewidth=linewidths[i],
				marker=markers[i], markersize=markersizes[i], markerfacecolor=markerfacecolors[i], 
				markeredgewidth=linewidths[i], label=labels[i])

	for k in range(len(refs_y)):
		ax.plot(refs_x[k], refs_y[k], linestyle='--', color='black')

	# Set figure properties
	taz.set_plot_properties(ax, **plot_properties)

	# Print some text
	plt.text(np.log(2.4), refs_y[0][-1], '$\mathcal{O}(\Delta t^2)$', horizontalalignment='left',
			 verticalalignment='center')
	#plt.text(np.log(2.4), refs_y[1][-1], '$\mathcal{O}(\Delta t^{2.5})$', horizontalalignment='left',
	#		 verticalalignment='center')
	plt.text(np.log(2.4), refs_y[1][-1], '$\mathcal{O}(\Delta t^3)$', horizontalalignment='left',
			 verticalalignment='center')

	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks([np.log(2.5), np.log(5), np.log(10), np.log(20)])
	ax2.set_xticklabels([2.5, 5.0, 10.0, 20.0])
	ax2.set_xlabel('$\\Delta t$ [s]')

	# Show
	fig.tight_layout()
	plt.show()

	#print(err[1])
	#print([np.log(err1/err2)/np.log(2) for err1, err2 in zip(err[1][:-1], err[1][1:])])
