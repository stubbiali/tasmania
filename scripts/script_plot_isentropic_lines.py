"""
A script to plot the isentropic lines, i.e., lines of constant potential temperature.
"""
import matplotlib as mpl
import matplotlib.animation as manimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_first_order_sedimentation_maccormack.pickle')
y_level = 0
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], 'results/figures/nmwc_model_check_maccormack_x_velocity')
fontsize         = 16
figsize          = [7,8]
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = [50,450]
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,8]
text			 = None #'$t = \dfrac{L}{\overline{u}}$'
text_loc		 = 'upper right'

#
# Plot
#
with open(filename, 'rb') as data:
	# Load data
	state_save = pickle.load(data)

	# Extract x-grid
	x = state_save.grid.x.values[:]

	# Extract height of the lines of constant potential temperature
	z_ = state_save['height'] if state_save['height'] is not None else state_save['height_on_interface_levels']
	z  = z_.values[:, y_level, :, time_level]
	nk = z.shape[1]

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes for visualization purposes
	x *= x_factor
	z *= z_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Plot the isolines
	for k in range(nk):
		ax.plot(x, z[:, k], color = 'gray', linewidth = 1)
	ax.plot(x, z[:, -1], color = 'black', linewidth = 1)

	# Attach labels to isolines
	plt.text(400, z[-10,-2], '$\\theta = 260$ K', horizontalalignment = 'center', verticalalignment = 'center',
			 bbox = dict(facecolor = 'white', edgecolor = 'white'))
	plt.text(400, z[-10,-7], '$\\theta = 265$ K', horizontalalignment = 'center', verticalalignment = 'center',
			 bbox = dict(facecolor = 'white', edgecolor = 'white'))
	plt.text(400, z[-10,-12], '$\\theta = 270$ K', horizontalalignment = 'center', verticalalignment = 'center',
			 bbox = dict(facecolor = 'white', edgecolor = 'white'))
	plt.text(400, z[-10,-17], '$\\theta = 275$ K', horizontalalignment = 'center', verticalalignment = 'center',
			 bbox = dict(facecolor = 'white', edgecolor = 'white'))
	plt.text(400, z[-10,-22], '$\\theta = 280$ K', horizontalalignment = 'center', verticalalignment = 'center',
			 bbox = dict(facecolor = 'white', edgecolor = 'white'))
	#plt.text(400, z[-10,-27], '$\\theta = 285$ K', horizontalalignment = 'center', verticalalignment = 'center',
	#		 bbox = dict(facecolor = 'white', edgecolor = 'white'))

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label)
	if x_lim is None:
		ax.set_xlim([x[0], x[-1]])
	else:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)

	# Show
	plt.show()
