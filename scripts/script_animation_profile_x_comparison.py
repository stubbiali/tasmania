"""
A script to generate an animation showing the time evolution of two fields along a cross line 
orthogonal to the :math:`yz`-plane.
"""
import matplotlib as mpl
import matplotlib.animation as manimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename    = [
			   os.path.join(os.environ['TASMANIA_ROOT'], 
			   				'data/verification_kessler_wrf_second_order_sedimentation_evaporation_leapfrog.pickle'),
			   os.path.join(os.environ['TASMANIA_ROOT'], 
			   				'data/verification_kessler_wrf_second_order_sedimentation_evaporation_maccormack.pickle'),
			   os.path.join(os.environ['TASMANIA_ROOT'], 
			   				'../kessler_wrf_saturation_sedimentation_evaporation_maccormack_reference.pickle'),
			   #os.path.join(os.environ['TASMANIA_ROOT'], 
			   #			'data/verification_kessler_wrf_second_order_sedimentation_evaporation_maccormack.pickle'),
			   ]
field       = [
			   'accumulated_precipitation',
			   'accumulated_precipitation',
			   'accumulated_precipitation',
			   'precipitation',
			  ]
y_level     = [0, 0, 0, 0]
z_level     = [-1, -1, -1, -1]
destination = os.path.join(os.environ['TASMANIA_ROOT'], 
						   '../foo.mp4')

#
# Animation settings
#
fontsize         = 16
figsize          = [7,8]
title            = 'Accumulated precipitation [mm]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
y_label			 = '' #'$z$ [km]'
y_lim            = [0,0.7]
field_factor     = [1., 1., 1., 1.]
color            = ['green', 'blue', 'black', 'blue']
linestyle		 = ['-', '-', '-', '-.']
linewidth        = [1.5, 1.5, 1.5, 1.5]
grid_on			 = True
fps				 = 6
legend			 = ['LF, $\Delta x = 5$ km', 'MC, $\Delta x = 5$ km', 'MC, $\Delta x = 1$ km', 'MC, evap. ON']
legend_location  = 'upper right'

#
# Load the fields
#
x   = []
var = []

for m in range(len(filename)):
	with open(filename[m], 'rb') as data:
		state_save = pickle.load(data)
		diagnostics_save = pickle.load(data)

		if state_save[field[m]] is not None: # The field to plot is a state variable
			time = state_save[field[m]].coords['time'].values 
			var_ = state_save[field[m]].values[:, y_level[m], z_level[m], :]
			var.append(var_)
			x_   = state_save.grid.x.values if var_.shape[0] == state_save.grid.nx \
				   else state_save.grid.x_at_u_locations.values
			x.append(x_)
		else:  # The field to plot is a diagnostic
			time = diagnostics_save[field[m]].coords['time'].values 
			var_ = diagnostics_save[field[m]].values[:, y_level[m], z_level[m], :]
			var.append(var_)
			x_   = diagnostics_save.grid.x.values if var_.shape[0] == diagnostics_save.grid.nx \
				   else diagnostics_save.grid.x_at_u_locations.values
			x.append(x_)

#
# Generate animation
#
utils_plot.animation_profile_x_comparison(time, x, var, destination,
										  fontsize   		= fontsize,
										  figsize    		= figsize,
										  title      		= title,
										  x_factor   		= x_factor,
										  x_label    		= x_label,
										  x_lim		 		= x_lim,
										  y_label	 		= y_label,
										  y_lim      		= y_lim,
										  field_factor		= field_factor,
										  color     		= color,
										  linestyle 		= linestyle, 
										  linewidth 		= linewidth,
										  grid_on    		= grid_on,
										  fps		 		= fps,
										  legend	 		= legend,
										  legend_location 	= legend_location,
										 )
