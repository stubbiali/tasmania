"""
A script to generate an animation showing the time evolution of a field along a line perpendicular to the :math:`yz`-plane.
"""
import os
import pickle

import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename    = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_sedimentation_maccormack.pickle')
field       = 'precipitation'
y_level     = 0
z_level     = -1
destination = os.path.join(os.environ['TASMANIA_ROOT'], '../foo.mp4')

#
# Optional settings
#
fontsize         = 16
figsize          = [7,8]
title            = 'Precipitation [mm h$^{-1}$]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
y_factor         = 1. #1.e-3
y_label			 = '' #'$z$ [km]'
y_lim            = None #[0,10]
color            = 'blue'
linewidth        = 1.
fps				 = 10
text			 = 'MC'
text_loc		 = 'upper right'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	diagnostics_save = pickle.load(data)

	# The field to plot is a state variable
	if False:
		state_save.animation_profile_x(field, y_level, z_level, destination, 
									   fontsize  = fontsize,
									   figsize   = figsize,
									   title     = title,
									   x_label   = x_label,
									   x_factor  = x_factor,
									   x_lim     = x_lim,
									   y_label   = y_label,
									   y_factor  = y_factor,
									   y_lim     = y_lim,
									   color     = color,
									   linewidth = linewidth,
									   fps       = fps,
									   text		 = text,
									   text_loc	 = text_loc,
									  )

	# The field to plot is a diagnostic
	if True:
		diagnostics_save.animation_profile_x(field, y_level, z_level, destination, 
									   fontsize  = fontsize,
									   figsize   = figsize,
									   title     = title,
									   x_label   = x_label,
									   x_factor  = x_factor,
									   x_lim     = x_lim,
									   y_label   = y_label,
									   y_factor  = y_factor,
									   y_lim     = y_lim,
									   color     = color,
									   linewidth = linewidth,
									   fps       = fps,
									   text		 = text,
									   text_loc	 = text_loc,
									  )
