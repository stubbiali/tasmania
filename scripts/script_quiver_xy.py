"""
A script to generate xy-quiver-plots.
"""
import os
import pickle

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_1_upwind.pickle')
field = 'horizontal_velocity'
z_level = -1
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180308_phd_meeting/img/verification_1_upwind_pressure_xy')
fontsize         = 16
figsize          = [8,8]
title            = ''
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None 
x_step			 = 2
y_factor         = 1.e-3
y_label			 = '$y$ [km]'
y_lim            = None 
y_step			 = 2
cmap_name        = 'BuRd' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 14
cbar_ticks_step  = 2
cbar_center      = 15.
cbar_half_width  = 6.5
cbar_x_label     = '$x$-velocity [m s$^{-1}$]'
cbar_y_label     = ''
cbar_title       = ''
cbar_orientation = 'horizontal'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)

	state_save.quiver_xy(field, z_level, time_level, 
						 show             = show,
						 destination      = destination,
						 fontsize         = fontsize,
						 figsize          = figsize,
						 title            = title, 
						 x_label          = x_label,
						 x_factor         = x_factor,
						 x_lim            = x_lim,
						 x_step           = x_step,
						 y_label          = y_label,
						 y_factor         = y_factor,
						 y_lim            = y_lim,
						 y_step           = y_step,
						 cmap_name        = cmap_name, 
						 cbar_levels      = cbar_levels,
						 cbar_ticks_step  = cbar_ticks_step,
						 cbar_center      = cbar_center, 
						 cbar_half_width  = cbar_half_width,
						 cbar_x_label     = cbar_x_label,
						 cbar_y_label     = cbar_y_label,
						 cbar_title       = cbar_title,
						 cbar_orientation = cbar_orientation,
						)
