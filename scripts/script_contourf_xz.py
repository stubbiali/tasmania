"""
A script to generate the contourf plot of a field at a cross section parallel to the :math:`xz`-plane.
"""
import numpy as np
import os
import pickle

import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_first_order_sedimentation_maccormack.pickle')
field = 'mass_fraction_of_precipitation_water_in_air'
y_level = 0
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], 'results/figures/nmwc_model_check_maccormack_x_velocity')
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
draw_z_isolines	 = True
cmap_name        = 'Blues' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 18
cbar_ticks_step  = 4
cbar_center      = None #15.
cbar_half_width  = None #11.
cbar_x_label     = ''
cbar_y_label     = ''
cbar_title       = ''
cbar_orientation = 'horizontal'
text			 = None #'$t = \dfrac{L}{\overline{u}}$'
text_loc		 = 'upper right'

#
# Plot
#
with open(filename, 'rb') as data:
	state_save = pickle.load(data)

	# Plot the specified field
	if True:
		state_save.contourf_xz(field, y_level, time_level, 
							   show            = show,
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
							   draw_z_isolines	= draw_z_isolines,
							   cmap_name        = cmap_name, 
							   cbar_levels      = cbar_levels,
							   cbar_ticks_step  = cbar_ticks_step,
							   cbar_center      = cbar_center, 
							   cbar_half_width  = cbar_half_width,
							   cbar_x_label     = cbar_x_label,
							   cbar_y_label     = cbar_y_label,
							   cbar_title       = cbar_title,
							   cbar_orientation = cbar_orientation,
							   text				= text,
							   text_loc			= text_loc,
							  )

	# Plot the analytical isothermal and isentropic flow over an isolated "Switch of Agnesi" mountain
	if False:
		grid = state_save.grid
		nx, nz = grid.nx, grid.nz
		xv = grid.x.values[:]
		x = np.repeat(xv[:, np.newaxis], nz+1, axis = 1)
		z = state_save['height'].values[:, 0, :, time_level]
		topography = z[:, 0]

		uex, wex = utils_meteo.get_isentropic_isothermal_analytical_solution(state_save.grid, 10., 250., 1., 1.e4,
												 	   						 x_staggered = False, z_staggered = True)

		utils_plot.contourf_xz(x, z, uex - 10., topography,
							   show			    = show,
							   destination		= destination,
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
							   draw_z_isolines	= draw_z_isolines,
							   cmap_name        = cmap_name, 
							   cbar_levels      = cbar_levels,
							   cbar_ticks_step  = cbar_ticks_step,
							   cbar_center      = cbar_center, 
							   cbar_half_width  = cbar_half_width,
							   cbar_x_label     = cbar_x_label,
							   cbar_y_label     = cbar_y_label,
							   cbar_title       = cbar_title,
							   cbar_orientation = cbar_orientation,
							   text				= text,
							   text_loc			= text_loc,
							   )

