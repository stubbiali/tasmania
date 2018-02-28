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
A script to generate xz-contourfs.
"""
import os
import pickle

import utils_meteo

#
# Mandatory settings
#
filename = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_2_maccormack.pickle')
field = 'x_velocity'
y_level = 0
time_level = -1

#
# Optional settings
#
ishow			 = True
destination		 = os.path.join(os.environ['TASMANIA_ROOT'], '../meetings/20180308_phd_meeting/img/isentropic_convergence_maccormack_u10_lx400_nz300_ray05_diff_8km_relaxed_horizontal_velocity_perturbation')
fontsize         = 16
figsize          = [7,8]
title            = ''
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,28]
field_factor     = 1.
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

	# Plot the specified field
	if True:
		state_save.contourf_xz(field, y_level, time_level, 
							   ishow            = ishow,
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

	# Plot the analytical isothermal and isentropic flow over an isolated "Switch of Agnesi" mountain
	if False:
		grid = state_save.grid
		topography = grid.topography_height[:, 0]
		height = state_save['height'].values[:, 0, :, time_level]

		uex, wex = utils_meteo.get_isothermal_solution(state_save.grid, 10., 250., 1., 1.e4,
												 	   x_staggered = False, z_staggered = False)

		utils_meteo.contourf_xz(grid, topography, height, 1.e4 * (uex - 10.), 
								show			 = show,
								destination		 = destination,
							    fontsize         = fontsize,
							   	figsize          = figsize,
								title            = title, 
								x_label          = x_label,
								x_factor         = x_factor,
								x_lim            = x_lim,
								z_label          = z_label,
								z_factor         = z_factor,
								z_lim            = z_lim,
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

