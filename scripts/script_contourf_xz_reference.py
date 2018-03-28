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
A script to generate xz-contourfs for outputs of the reference isentropic model.
"""
import numpy as np
import os
import pickle

from tasmania.grids.grid_xyz import GridXYZ
import tasmania.utils.utils_meteo as utils_meteo
import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename   = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_leapfrog_reference.npz')
field      = 'horizontal_velocity' #'specific_humidity' #'specific_cloud_liquid_water_content'
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
x_lim			 = None #[-40,40]
z_factor         = 1.e-3
z_label			 = '$z$ [km]'
z_lim            = [0,10]
field_factor     = 1. #1.e3
plot_height		 = True
cmap_name        = 'BuRd' # Alternatives: Blues, BuRd, jet, RdBu, RdYlBu, RdYlGn
cbar_levels      = 18
cbar_ticks_step  = 4
cbar_center      = 15.
cbar_half_width  = None #0.85
cbar_x_label     = 'Water vapor [g kg$^{-1}$]'
cbar_y_label     = ''
cbar_title       = ''
cbar_orientation = 'horizontal'
text			 = None #'$t = \dfrac{L}{\overline{u}}$'
text_loc		 = 'upper right'

#
# Plot
#
with np.load(filename) as data:
	# Extract the height
	height = data['height'][time_level,::-1,:].transpose()

	# Extract the field to plot
	field = data[field][time_level,::-1,:].transpose()

	# Create the underlying grid
	x, nx = data['x'], data['nx']
	nz = data['nz']
	th00, thll = data['th00'][0], data['thl'][0]
	topo_max, topo_width = data['topomx'], data['topowd']
	grid = GridXYZ(domain_x        = [x[0], x[-1]], 
				   nx              = nx, 
				   domain_y        = [-1., 1.], 
				   ny              = 1, 
				   domain_z        = [th00 + thll, th00], 
				   nz              = nz,
				   units_x         = 'm', 
				   dims_x          = 'x', 
				   units_y         = 'm', 
				   dims_y          = 'y', 
				   units_z         = 'K', 
				   dims_z          = 'air_potential_temperature', 
				   z_interface     = None,
				   topo_type       = 'gaussian',  
				   topo_max_height = topo_max,
				   topo_width_x    = topo_width)
	
	# Plot the specified field
	utils_plot.contourf_xz(grid, grid.topography_height[:,0], height, field, 
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
						   text_loc			= text_loc)

