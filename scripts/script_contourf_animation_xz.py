"""
A script to generate a xz-contourf animation of a field.
"""
import os
import pickle

import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename    = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_maccormack_animation.pickle')
field       = 'mass_fraction_of_cloud_liquid_water_in_air'
y_level     = 0
destination = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180328_status_meeting/movie/verification_kessler_wrf_maccormack_x_cloud_liquid_water.mp4')

#
# Optional settings
#
fontsize         = 16
figsize          = [7,8]
title            = 'Cloud liquid water [g kg$^{-1}$]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,10]
field_factor     = 1.e3
plot_height		 = True
cmap_name        = 'Blues' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 18
cbar_ticks_step  = 4
cbar_center      = None #15.
cbar_half_width  = None #9.5
cbar_x_label     = None
cbar_y_label     = None
cbar_title       = None
cbar_orientation = 'horizontal'
fps				 = 10
text			 = None #'$t = \dfrac{L}{\overline{u}}$'
text_loc		 = 'upper right'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)

	# Plot the specified field
	state_save.contourf_animation_xz(field, y_level, destination, 
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
								     plot_height	  = plot_height,
								     cmap_name        = cmap_name, 
								     cbar_levels      = cbar_levels,
								     cbar_ticks_step  = cbar_ticks_step,
								     cbar_center      = cbar_center, 
								     cbar_half_width  = cbar_half_width,
								     cbar_x_label     = cbar_x_label,
								     cbar_y_label     = cbar_y_label,
								     cbar_title       = cbar_title,
								     cbar_orientation = cbar_orientation,
									 fps              = fps,
								     text			  = text,
								     text_loc		  = text_loc,
								    )
