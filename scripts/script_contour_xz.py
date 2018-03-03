"""
A script to generate a xz-contour of a field.
"""
import os
import pickle

import utils.utils_meteo as utils_meteo

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_moist_leapfrog.pickle')
field = 'water_vapor'
y_level = 0
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180308_phd_meeting/img/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_8km_relaxed_horizontal_velocity_perturbation')
fontsize         = 16
figsize          = [7,8]
title            = ''
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = None #[0,10]
field_factor     = 1.
plot_height		 = True
text			 = None
text_loc		 = 'upper right'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)

	state_save.contour_xz(field, y_level, time_level, 
						  show             = show,
						  destination      = destination,
						  fontsize         = fontsize,
						  figsize          = figsize,
						  title            = title, 
						  x_label          = x_label,
						  x_factor         = x_factor,
						  x_lim            = x_lim,
						  z_label          = z_label,
						  z_factor         = z_factor,
						  z_lim            = z_lim,
						  field_factor     = field_factor,
						  plot_height	   = plot_height,
						  text			   = text,
						  text_loc		   = text_loc,
						 )
