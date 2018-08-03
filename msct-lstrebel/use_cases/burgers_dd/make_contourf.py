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
Utility to generate the contourf plot of a certain field.
"""
import matplotlib as mpl
import matplotlib.animation as manimation
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import pickle

import utils

#
# User-defined settings:
# filename 				Path to the location where the solution is stored
# field 				Field to plot; options: 'u', 'v', 'speed', 'u_zhao', 'v_zhao'
# time_level			The time level to plot
# fontsize         		Font size
# figsize			 	Figure size
# title            		Figure title
# x_label          		Label for the x-axis
# x_factor         		Multiplying factor for the x-axis
# x_lim			 		Limits for the x-axis
# y_label          		Label for the y-axis
# y_factor         		Multiplying factor for the y-axis
# y_lim			 		Limits for the y-axis
# field_factor     		Multiplying factor for the field
# cmap_name        		Name of the colormap
# cbar_levels      		Number of levels of the colorbar
# cbar_ticks_step  		Ticks frequency for the colorbar; if 1, each tick is drawn
# cbar_center      		Center of the spectrum spanned by the colorbar
# cbar_half_width  		Half-width of the spectrum spanned by the colorbar
# cbar_x_label     		Label for the x-axis of the colorbar
# cbar_y_label     		Label for the y-axis of the colorbar
# cbar_title       		Title of the colorbar
# cbar_orientation 		Orientation of the colorbar; options: 'vertical', 'horizontal'
# text			 		Text to anchor to the figure
# text_loc		 		Text location
#
testname = "test_zhao_forward_backward" #'test_shankar_forward_backward'
filename 			= testname +'.pickle'
field 				= 'u'
time_level			= -1
fontsize         	= 16
figsize			 	= [7,8]
title            	= '$x$-velocity [m s$^{-1}$]'
x_label          	= '$x$'
x_factor         	= 1.
x_lim			 	= [0,1]
y_label          	= '$y$'
y_factor         	= 1.
y_lim			 	= [0,1]
field_factor     	= 1.
cmap_name        	= 'BuRd'
cbar_levels      	= 18
cbar_ticks_step  	= 4
cbar_center      	= None
cbar_half_width  	= None
cbar_x_label     	= ''
cbar_y_label     	= ''
cbar_title       	= ''
cbar_orientation 	= 'horizontal'
text			 	= None
text_loc		 	= 'upper right'

#
# Utility
#
# Load data
with open(filename, 'rb') as data:
	t   = pickle.load(data)
	x   = pickle.load(data)
	y   = pickle.load(data)
	u   = pickle.load(data)
	v   = pickle.load(data)
	eps = pickle.load(data)

# Shortcuts
nx, ny, _ = u.shape

# Global settings
mpl.rcParams['font.size'] = fontsize

# Instantiate figure and axis objects
fig, ax = plt.subplots(figsize = figsize)

# Construct grid
x = np.repeat(x[:, np.newaxis], ny, axis = 1)
y = np.repeat(y[np.newaxis, :], nx, axis = 0)

# The field to plot
if field == 'u':
	var = u[:,:,time_level]
elif field == 'v':
	var = v[:,:,time_level]
elif field == 'speed':
	var = np.sqrt(u ** 2 + v ** 2)[:,:,time_level]
elif field == 'u_zhao':
	t_ = t[time_level].seconds if t[time_level].seconds > 0 else t[time_level].microseconds / 1.e6
	var = - 2. * eps * 2. * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t_) * np.cos(2. * np.pi * x) * np.sin(np.pi * y) / \
		  (2. + np.exp(- 5. * np.pi * np.pi * eps * t_) * np.sin(2. * np.pi * x) * np.sin(np.pi * x))
elif field == 'v_zhao':
	t_ = t[time_level].seconds if t[time_level].seconds > 0 else t[time_level].microseconds / 1.e6
	var = - 2. * eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t_) * np.sin(2. * np.pi * x) * np.cos(np.pi * y) / \
		  (2. + np.exp(- 5. * np.pi * np.pi * eps * t_) * np.sin(2. * np.pi * x) * np.sin(np.pi * x))

# Rescale the axes and the field for visualization purposes
x   *= x_factor
y   *= y_factor
var	*= field_factor

# Create the colorbar for the colormap
field_min, field_max = np.amin(var), np.amax(var)
if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
	cbar_lb, cbar_ub = field_min, field_max
else:
	half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None \
				 else cbar_half_width
	cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

# Create the colormap
if cmap_name == 'BuRd':
	cm = utils.reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
else:
	cm = plt.get_cmap(cmap_name)

# Plot the field
surf = plt.contourf(x, y, var, color_scale, cmap = cm)

# Plot without interpolation artifacts (only for Shakar IC)
#surf = plt.imshow(var.T, extent=(0,2,0,2), origin='lower', cmap=cm, interpolation='none')

# Set axis labels
ax.set(xlabel = x_label, ylabel = y_label)

# Set limits for x-axis
if x_lim is None:
	ax.set_xlim([x[0,0], x[-1,0]])
else:
	ax.set_xlim(x_lim)

# Set limits for y-axis
if y_lim is None:
	ax.set_ylim([y[0,0], y[0,-1]])
else:
	ax.set_ylim(y_lim)

# Set colorbar
cb = plt.colorbar(orientation = cbar_orientation)
cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
if cbar_title is not None:
	cb.ax.set_title(cbar_title)
if cbar_x_label is not None:
	cb.ax.set_xlabel(cbar_x_label)
if cbar_y_label is not None:
	cb.ax.set_ylabel(cbar_y_label)

# Add text
if text is not None:
	ax.add_artist(AnchoredText(text, loc = text_loc))

# Add title
plt.title(title, loc = 'left', fontsize = fontsize - 1)
plt.title(str(t[time_level]), loc = 'right', fontsize = fontsize - 1)

plt.savefig(testname + "_field_" + field + "_at_" + str(t[time_level]) +".png")
# Show
#plt.show()
