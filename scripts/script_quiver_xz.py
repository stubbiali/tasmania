"""
A script to generate the quiver plot of a vector field at a cross section parallel to the :math:`xz`-plane.
"""
import os
import pickle

#
# Mandatory settings
#
filename   = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_second_order_sedimentation_evaporation_maccormack.pickle')
field      = 'velocity'
y_level    = 0
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180308_phd_meeting/img/verification_1_upwind_pressure_xy')
fontsize         = 16
figsize          = [8,8]
title            = 'Velocity field'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = [0,500]
x_step			 = 6
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,10] 
z_step			 = 1
cmap_name        = 'BuRd' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 14
cbar_ticks_step  = 2
cbar_center      = 15.
cbar_half_width  = None #6.5
cbar_x_label     = ''
cbar_y_label     = ''
cbar_title       = ''
cbar_orientation = 'horizontal'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)

	state_save.quiver_xz(field, y_level, time_level, 
						 show             = show,
						 destination      = destination,
						 fontsize         = fontsize,
						 figsize          = figsize,
						 title            = title, 
						 x_label          = x_label,
						 x_factor         = x_factor,
						 x_lim            = x_lim,
						 x_step           = x_step,
						 z_label          = z_label,
						 z_factor         = z_factor,
						 z_lim            = z_lim,
						 z_step           = z_step,
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
