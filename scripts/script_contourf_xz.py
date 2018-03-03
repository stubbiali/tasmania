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
A script to generate a xz-contourf of a field.
"""
import os
import pickle

import utils.utils_meteo as utils_meteo
import utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_moist_advection_upwind.pickle')
field = 'water_vapor'
y_level = 0
time_level = 4

#
# Optional settings
#
show			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180308_phd_meeting/img/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_8km_relaxed_horizontal_velocity_perturbation')
fontsize         = 16
figsize          = [9,7]
title            = ''
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,20]
field_factor     = 1.e3
plot_height		 = True
cmap_name        = 'Blues' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 18
cbar_ticks_step  = 4
cbar_center      = 0.5 * 23
cbar_half_width  = 0.5 * 23
cbar_x_label     = ''
cbar_y_label     = 'Water vapor [g kg$^{-1}$]'
cbar_title       = ''
cbar_orientation = 'vertical'
text			 = '$t = \dfrac{L}{\overline{u}}$'
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
							   plot_height		= plot_height,
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
							   plot_height		= plot_height,
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

