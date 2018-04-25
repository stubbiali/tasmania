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
A script to generate an animation showing the time evolution of two fields along a cross line 
orthogonal to the :math:`yz`-plane.
"""
import matplotlib as mpl
import matplotlib.animation as manimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import tasmania.utils.utils_plot as utils_plot

#
# Mandatory settings
#
filename    = [
			   os.path.join(os.environ['TASMANIA_ROOT'], 
			   				'data/verification_kessler_wrf_second_order_sedimentation_evaporation_maccormack.pickle'),
			   os.path.join(os.environ['TASMANIA_ROOT'], 
			   				'data/slow_tendency_kessler_wrf_sedimentation_evaporation_maccormack.pickle'),
			   #os.path.join(os.environ['TASMANIA_ROOT'], 
			   #			'data/verification_kessler_wrf_second_order_sedimentation_maccormack.pickle'),
			   #os.path.join(os.environ['TASMANIA_ROOT'], 
			   #			'data/verification_kessler_wrf_second_order_sedimentation_evaporation_maccormack.pickle'),
			   ]
field       = [
			   'precipitation',
			   'precipitation',
			   'precipitation',
			   'precipitation',
			  ]
y_level     = [0, 0, 0, 0]
z_level     = [-1, -1, -1, -1]
destination = os.path.join(os.environ['TASMANIA_ROOT'], 
						   'results/movies/kessler_sedimentation_evaporation/kessler_sedimentation_evaporation_slow_tendency_adjustment_precipitation.mp4')

#
# Animation settings
#
fontsize         = 16
figsize          = [7,8]
title            = 'Precipitation [mm h$^{-1}$]'
x_factor         = 1.e-3
x_label          = '$x$ [km]'
x_lim			 = None #[-40,40]
y_label			 = '' #'$z$ [km]'
y_lim            = [0,0.7]
field_factor     = [1., 1., 1., 1.]
color            = ['blue', 'orange', 'blue', 'blue']
linestyle		 = ['-', '--', '-', '-.']
linewidth        = [1.5, 1.5, 1.5, 1.5]
grid_on			 = True
fps				 = 6
legend			 = ['MC ADJ', 'MC ST', 'MC, evap. OFF', 'MC, evap. ON']
legend_location  = 'upper right'

#
# Load the fields
#
for m in range(len(filename)):
	with open(filename[m], 'rb') as data:
		state_save = pickle.load(data)
		diagnostics_save = pickle.load(data)

		if state_save[field[m]] is not None: # The field to plot is a state variable
			if m == 0:
				time = state_save[field[m]].coords['time'].values 
				var_ = state_save[field[m]].values[:, y_level[m], z_level[m], :]
				var  = var_[np.newaxis, :, :]
				x_   = state_save.grid.x.values if var_.shape[0] == state_save.grid.nx \
				 	   else state_save.grid.x_half_levels.values
				x    = x_[np.newaxis, :]
			else:
				var_ = state_save[field[m]].values[:, y_level[m], z_level[m], :]
				var  = np.concatenate((var, var_[np.newaxis, :, :]), axis = 0)
				x_   = state_save.grid.x.values if var_.shape[0] == state_save.grid.nx \
				 	   else state_save.grid.x_half_levels.values
				x    = np.concatenate((x, x_[np.newaxis, :]), axis = 0)
		else:  # The field to plot is a diagnostic
			if m == 0:
				time = diagnostics_save[field[m]].coords['time'].values 
				var_ = diagnostics_save[field[m]].values[:, y_level[m], z_level[m], :]
				var  = var_[np.newaxis, :, :]
				x_   = diagnostics_save.grid.x.values if var_.shape[0] == diagnostics_save.grid.nx \
				 	   else diagnostics_save.grid.x_half_levels.values
				x    = x_[np.newaxis, :]
			else:
				var_ = diagnostics_save[field[m]].values[:, y_level[m], z_level[m], :]
				var  = np.concatenate((var, var_[np.newaxis, :, :]), axis = 0)
				x_   = diagnostics_save.grid.x.values if var_.shape[0] == diagnostics_save.grid.nx \
				 	   else diagnostics_save.grid.x_half_levels.values
				x    = np.concatenate((x, x_[np.newaxis, :]), axis = 0)

#
# Generate animation
#
utils_plot.animation_profile_x_comparison(time, x, var, destination,
										  fontsize   		= fontsize,
										  figsize    		= figsize,
										  title      		= title,
										  x_factor   		= x_factor,
										  x_label    		= x_label,
										  x_lim		 		= x_lim,
										  y_label	 		= y_label,
										  y_lim      		= y_lim,
										  field_factor		= field_factor,
										  color     		= color,
										  linestyle 		= linestyle, 
										  linewidth 		= linewidth,
										  grid_on    		= grid_on,
										  fps		 		= fps,
										  legend	 		= legend,
										  legend_location 	= legend_location,
										 )
