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
filename1    = os.path.join(os.environ['TASMANIA_ROOT'], 
							'data/verification_kessler_wrf_sedimentation_leapfrog.pickle')
field1       = 'accumulated_precipitation'
filename2    = os.path.join(os.environ['TASMANIA_ROOT'], 
							'data/verification_kessler_wrf_sedimentation_evaporation_leapfrog.pickle')
field2       = 'accumulated_precipitation'
y_level     = 0
z_level     = -1
destination = os.path.join(os.environ['TASMANIA_ROOT'], 
						   'results/movies/kessler_sedimentation_evaporation/verification_kessler_wrf_sedimentation_evaporation_leapfrog_accumulated_precipitation.mp4')

#
# Animation settings
#
fontsize         = 16
figsize          = [7,8]
title            = 'Accumulated precipitation [mm]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
y_factor1        = 1. #1.e-3
y_factor2        = 1. #1.e-3
y_label			 = '' #'$z$ [km]'
y_lim            = [0,1.2]
color1           = 'green'
linestyle1		 = '-'
linewidth1       = 1.5
color2           = 'green'
linestyle2		 = '--' 
linewidth2       = 1.5
grid_on			 = True
fps				 = 7
legend1			 = 'Rain evap. OFF'
legend2			 = 'Rain evap. ON'
legend_location  = 'upper right'

#
# Load first field
#
with open(filename1, 'rb') as data:
	state_save = pickle.load(data)
	diagnostics_save = pickle.load(data)

	if False: # The field to plot is a state variable
		time = state_save[field1].coords['time'].values 
		var1 = state_save[field1].values[:, y_level, z_level, :]
		x1 = state_save.grid.x.values if var1.shape[0] == state_save.grid.nx \
			 else state_save.grid.x_half_levels.values

	if True:  # The field to plot is a diagnostic
		time = diagnostics_save[field1].coords['time'].values 
		var1 = diagnostics_save[field1].values[:, y_level, z_level, :]
		x1 = diagnostics_save.grid.x.values if var1.shape[0] == diagnostics_save.grid.nx \
			 else diagnostics_save.grid.x_half_levels.values

#
# Load second field
#
with open(filename2, 'rb') as data:
	state_save = pickle.load(data)
	diagnostics_save = pickle.load(data)

	if False: # The field to plot is a state variable
		var2 = state_save[field2].values[:, y_level, z_level, :]
		x2 = state_save.grid.x.values if var2.shape[0] == state_save.grid.nx \
			 else state_save.grid.x_half_levels.values

	if True:  # The field to plot is a diagnostic
		var2 = diagnostics_save[field2].values[:, y_level, z_level, :]
		x2 = diagnostics_save.grid.x.values if var2.shape[0] == diagnostics_save.grid.nx \
			 else diagnostics_save.grid.x_half_levels.values

#
# Generate animation
#
utils_plot.animation_profile_x_comparison(time, x1, var1, x2, var2, destination,
										  fontsize   		= fontsize,
										  figsize    		= figsize,
										  title      		= title,
										  x_factor   		= x_factor,
										  x_label    		= x_label,
										  x_lim		 		= x_lim,
										  y_factor1  		= y_factor1,
										  y_factor2  		= y_factor2,
										  y_label	 		= y_label,
										  y_lim      		= y_lim,
										  color1     		= color1,
										  linestyle1 		= linestyle1, 
										  linewidth1 		= linewidth1,
										  color2     		= color2,
										  linestyle2 		= linestyle2, 
										  linewidth2 		= linewidth2,
										  grid_on    		= grid_on,
										  fps		 		= fps,
										  legend1	 		= legend1,
										  legend2	 		= legend2,
										  legend_location 	= legend_location,
										 )
