# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
A script to generate the contourf plot of a field at a cross section parallel to the :math:`xz`-plane.
"""
import os
import pickle

import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_first_order_sedimentation_maccormack.pickle')
field = 'x_velocity'
y_level = 0
time_level = -1

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], 'results/figures/nmwc_model_check_maccormack_x_velocity')
fontsize         = 16
figsize          = [8,7]
title            = '$x$-velocity [m s$^{-1}$]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,10]
field_factor     = 1.
draw_z_isolines	 = True
cmap_name        = 'BuRd' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 12
cbar_ticks_step  = 1
cbar_center      = 15.
cbar_half_width  = 11.
cbar_x_label     = ''
cbar_y_label     = ''
cbar_title       = ''
cbar_orientation = 'vertical'
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
		topography = grid.topography_height[:, 0]
		height = state_save['height'].values[:, 0, :, time_level]

		uex, wex = utils_meteo.get_isentropic_isothermal_analytical_solution(state_save.grid, 10., 250., 1., 1.e4,
												 	   						 x_staggered = False, z_staggered = False)

		utils_plot.contourf_xz(grid, topography, height, uex - 10., 
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
							   field_factor     = 1.e-4,
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

