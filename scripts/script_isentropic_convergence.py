"""
A script to test the convergence of numerical schemes for the isentropic model.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import utils_meteo as utils

#
# Mandatory settings
#
ihorizontal = True
ishow = False
destination = os.path.join(os.environ['TASMANIA_ROOT'], 
						   '../meetings/20180308_phd_meeting/img/isentropic_convergence_upwind_horizontal_velocity_error_1')

keys = ['upwind', 'maccormack']
references = \
	{
	 keys[0]: os.path.join(os.environ['TASMANIA_ROOT'], 
	 					   'data/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_05km_relaxed.pickle'),
	 keys[1]: os.path.join(os.environ['TASMANIA_ROOT'], 
	 					   'data/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_05km_relaxed.pickle'),
	}
filenames = \
	{
	 keys[0]: [os.path.join(os.environ['TASMANIA_ROOT'], 
							'data/isentropic_convergence_upwind_u10_lx400_nz300_ray05_') + suffix \
			   for suffix in ['8km_relaxed.pickle']],
	 keys[1]: [os.path.join(os.environ['TASMANIA_ROOT'], 
							'data/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_') + suffix \
			   for suffix in ['8km_relaxed.pickle']],
	}
sampling_steps = \
	{
	 keys[0]: [16, 8, 4, 2, 1],
	 keys[1]: [16, 8, 4, 2],
	}
x_velocity_initial = \
	{
	 keys[0]: 10.,
	 keys[1]: 10.,
	}
temperature_initial = \
	{
     keys[0]: 250.,
     keys[1]: 250.,
	}
mountain_height = \
	{
     keys[0]: 1,
     keys[1]: 1,
	}
mountain_width = \
	{
     keys[0]: 1.e4,
     keys[1]: 1.e4,
	}
color = \
	{
	 keys[0]: 'red',
	 keys[1]: 'blue',
	}
marker = \
	{
	 keys[0]: 's',
	 keys[1]: 'o',
	}
markersize = \
	{
	 keys[0]: 8,
	 keys[1]: 8,
	}
linestyle = \
	{
	 keys[0]: '-',
	 keys[1]: '-',
	}
linewidth = \
	{
	 keys[0]: 1.5,
	 keys[1]: 1.5,
	}
label = \
	{
	 keys[0]: 'Upwind',
	 keys[1]: 'MacCormack',
	}

#
# Optional settings
#
figsize         = [8,8]
title           = ''
x_factor        = 1.e-3
x_label         = '$\Delta x$ [km]'
x_lim           = [8./13.*0.5, 13]
x_ticks         = np.arange(1,9)
y_factor        = 1.e-3
y_label			= 'RMSE [m s$^{-1}$]'
y_lim           = [7e-5, 1e-2]

#
# Run
#
mpl.rcParams['font.size'] = 16

fig, ax = plt.subplots(figsize = figsize)

for key in keys[:1]:
	reference			 = references[key]
	filenames_list       = filenames[key]
	step				 = sampling_steps[key]
	x_velocity_initial_  = x_velocity_initial[key]
	temperature_initial_ = temperature_initial[key]
	mountain_height_     = mountain_height[key]
	mountain_width_      = mountain_width[key]
	color_               = color[key]
	marker_              = marker[key]
	markersize_          = markersize[key]
	linestyle_			 = linestyle[key]
	linewidth_			 = linewidth[key]
	label_				 = label[key]

	dx = np.zeros(len(filenames_list), dtype = float)
	e  = np.zeros(len(filenames_list), dtype = float)

	# Load reference solution
	with open(reference, 'rb') as data:
		state_save = pickle.load(data)

		u_ref = state_save['x_momentum_isentropic'].values[320:481, 0, :, -1] / \
				state_save['isentropic_density'].values[320:481, 0, :, -1]

		height = 0.5 * (state_save['height'].values[320:481, 0, :-1, -1] +
						state_save['height'].values[320:481, 0, 1:, -1])
		h = 0.5 * (height[:-1, :] + height[1:, :])
		h_ref = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)
		w_ref = u_ref * (h_ref[1:, :] - h_ref[:-1, :]) / state_save.grid.dx

	for i in range(len(filenames_list)):
		with open(filenames_list[i], 'rb') as data:
			state_save = pickle.load(data)
			
			grid = state_save.grid
			dx[i] = x_factor * grid.dx

			height = state_save['height'].values[:, 0, :, -1]
			ni_start = grid.nx - np.sum(grid.x.values >= -40.e3) 
			ni_stop = np.sum(grid.x.values <= 40.e3)
			nk = np.sum(height[int(0.5 * (ni_start + ni_stop)), :] <= 7500.)

			if ihorizontal:
				topography = grid.topography_height[:, 0]
				uex, wex = utils.get_isothermal_solution(grid, x_velocity_initial_, temperature_initial_, 
													 mountain_height_, mountain_width_,
												 	 x_staggered = False, z_staggered = False)

				u = state_save['x_momentum_isentropic'].values[ni_start:ni_stop, 0, nk:, -1] / \
					state_save['isentropic_density'].values[ni_start:ni_stop:, 0, nk:, -1]
	
				#e[i] = np.sqrt(1. / ((ni_stop - ni_start) * nk) * np.linalg.norm(u - uex[ni_start:ni_stop, -nk:]) ** 2)
				e[i] = np.sqrt(1. / ((ni_stop - ni_start) * (grid.nz - nk)) * \
							   np.sum((u[:, -nk:] - u_ref[::step[i], -nk:]) * (u[:, -nk:] - u_ref[::step[i], -nk:])))
			else:
				topography = grid.topography_height[:, 0]
				uex, wex = utils.get_isothermal_solution(grid, x_velocity_initial_, temperature_initial_, 
													 mountain_height_, mountain_width_,
												 	 x_staggered = True, z_staggered = False)

				u = state_save['x_momentum_isentropic'].values[ni_start:ni_stop, 0, :, -1] / \
					state_save['isentropic_density'].values[ni_start:ni_stop, 0, :, -1]

				h = 0.25 * (height[ni_start:ni_stop-1, :-1] + height[ni_start+1:ni_stop, :-1] + \
							height[ni_start:ni_stop-1, 1:] + height[ni_start+1:ni_stop, 1:])
				h = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)
				w = u * (h[1:, :] - h[:-1, :]) / state_save.grid.dx

				e[i] = np.sqrt(1. / ((ni_stop - ni_start) * (grid.nz - nk)) * \
						   			  np.sum((w[:, -nk:] - w_ref[::step[i], -nk:]) * 
									  		 (w[:, -nk:] - w_ref[::step[i], -nk:])))

	ax.loglog(dx, e, color = color_, marker = marker_, markersize = markersize_, 
					 markerfacecolor = 'None', markeredgecolor = color_, markeredgewidth = linewidth_,
	                 linestyle = linestyle_, linewidth = linewidth_, basex = 2, label = label_)

ax.loglog([], [], color = 'blue', marker = 'o', markersize = 8, 
				 markerfacecolor = 'None', markeredgecolor = 'blue', markeredgewidth = 1.5,
				 linestyle = '-', linewidth = 1.5, basex = 2, label = 'MacCormack')

ax.set(title = title)
plt.grid(True)
	
ax.set(xlabel = x_label)
if x_lim is not None:
	ax.set_xlim(x_lim)
plt.xticks(x_ticks)
ax.invert_xaxis()

ax.set(ylabel = y_label)
if y_lim is not None:
	ax.set_ylim(y_lim)

#
# Extra plottings
#
if True:
	ax.loglog(np.array([0.5, 1., 2., 4.]), 2e-3 * np.array([0.5, 1., 2., 4.]), 
			  color = 'black', linestyle = '--', basex = 2)
	plt.text(0.5, 1e-3, '  $\mathcal{O}(\Delta x$)', horizontalalignment = 'left', verticalalignment = 'bottom')
	ax.loglog(np.array([1., 2., 4.]), 1e-4 * np.array([1., 4., 16.]), 
			  color = 'black', linestyle = '--', basex = 2)
	plt.text(1, 1e-4, '  $\mathcal{O}(\Delta x^2$)', horizontalalignment = 'left', verticalalignment = 'top')

ax.legend()
plt.legend(frameon = True)

if ishow or (destination is None):
	plt.show()
else:
	plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

