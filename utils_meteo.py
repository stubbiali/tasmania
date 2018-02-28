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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from namelist import cp, datatype, g, p_ref, Rd
from utils import reverse_colormap
from utils import smaller_than as lt

def contourf_xz(grid, topography, height, field, **kwargs):
	# Shortcuts
	nx, nz = grid.nx, grid.nz
	ni, nk = field.shape

	# Get keyword arguments
	ishow            = kwargs.get('ishow', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', '{} [${}$]'.format(grid.x.dims, grid.x.attrs.get('units', '')))
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', '{} [${}$]'.format(grid.z.dims, grid.z.attrs.get('units', '')))
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the field for visualization purposes
	field *= field_factor

	# The x-grid underlying the isentropes and the field
	x1 = x_factor * np.repeat(grid.x.values[:, np.newaxis], nz, axis = 1)
	xv = x_factor * (grid.x.values if ni == nx else grid.x_half_levels.values)
	x2 = np.repeat(xv[:, np.newaxis], nk, axis = 1)

	# The isentropes
	z = z_factor * height
	z1 = z if nk == nz + 1 else 0.5 * (z[:, :-1] + z[:, 1:])

	# The z-grid underlying the field
	z2 = np.zeros((ni, nk), dtype = datatype)
	if ni == nx:
		z2[:, :] = z1[:, :]
	else:
		z2[1:-1, :] = 0.5 * (z1[:-1, :] + z1[1:, :])
		z2[0, :], z2[-1, :] = z2[1, :], z2[-2, :]

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Plot the isentropes
	for k in range(0, nk):
		ax.plot(x1[:, 0], z1[:, k], color = 'gray', linewidth = 1)
	ax.plot(x1[:, 0], z[:, -1], color = 'black', linewidth = 1)

	# Create colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	if cmap_name == 'BuRd':
		cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the field
	surf = plt.contourf(x2, z2, field, color_scale, cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label, title = title)
	if x_lim is None:
		ax.set_xlim([x1[0,0], x1[-1,0]])
	else:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)
	
	# Set colorbar
	cb = plt.colorbar(orientation = cbar_orientation)
	cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if ishow or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def contourf_xy(grid, field, **kwargs):
	# Shortcuts
	nx, ny = grid.nx, grid.ny
	ni, nj = field.shape

	# Get keyword arguments
	ishow            = kwargs.get('ishow', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', '{} [${}$]'.format(grid.x.dims, grid.x.attrs.get('units', '')))
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	y_label          = kwargs.get('y_label', '{} [${}$]'.format(grid.y.dims, grid.y.attrs.get('units', '')))
	y_factor         = kwargs.get('y_factor', 1.)
	y_lim			 = kwargs.get('y_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the field for visualization purposes
	field *= field_factor

	# The grid
	xv = x_factor * grid.x.values if ni == nx else x_factor * grid.x_half_levels.values
	yv = y_factor * grid.y.values if nj == ny else y_factor * grid.y_half_levels.values
	x, y = np.meshgrid(xv, yv, indexing = 'ij')

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Draw topography isolevels
	cs = plt.contour(x, y, grid._topography._topo_final, colors = 'gray')

	# Create colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	if cmap_name == 'BuRd':
		cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the field
	plt.contourf(x, y, field, color_scale, cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = y_label, title = title)
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	
	# Set colorbar
	cb = plt.colorbar(orientation = cbar_orientation)
	cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if ishow or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def quiver_xy(grid, vx, vy, scalar, **kwargs):
	# Shortcuts
	nx, ny = grid.nx, grid.ny
	ni, nj = scalar.shape

	# Get keyword arguments
	ishow            = kwargs.get('ishow', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', '{} [${}$]'.format(grid.x.dims, grid.x.attrs.get('units', '')))
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	x_step           = kwargs.get('x_step', 2)
	y_label          = kwargs.get('y_label', '{} [${}$]'.format(grid.y.dims, grid.y.attrs.get('units', '')))
	y_factor         = kwargs.get('y_factor', 1.)
	y_lim			 = kwargs.get('y_lim', None)
	y_step           = kwargs.get('y_step', 2)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# The grid
	xv = x_factor * grid.x.values if ni == nx else x_factor * grid.x_half_levels.values
	yv = y_factor * grid.y.values if nj == ny else y_factor * grid.y_half_levels.values
	x, y = np.meshgrid(xv, yv, indexing = 'ij')

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Draw topography isolevels
	plt.contour(x, y, grid._topography._topo_final, colors = 'gray')

	# Create colormap
	scalar_min, scalar_max = np.amin(scalar), np.amax(scalar)
	if cbar_center is None or not (lt(scalar_min, cbar_center) and lt(cbar_center, scalar_max)):
		cbar_lb, cbar_ub = scalar_min, scalar_max
	else:
		half_width = max(cbar_center - scalar_min, scalar_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	if cmap_name == 'BuRd':
		cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Generate quiver-plot
	q = plt.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step], vx[::x_step, ::y_step], vy[::x_step, ::y_step]) 
				   #scalar[::x_step, ::y_step], cmap = cm, units = 'width')

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = y_label, title = title)
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	
	# Set colorbar
	#cb = plt.colorbar(orientation = cbar_orientation)
	#cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	#cb.ax.set_title(cbar_title)
	#cb.ax.set_xlabel(cbar_x_label)
	#cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if ishow or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def get_isothermal_solution(grid, x_velocity_initial, temperature, mountain_height, mountain_width,
							x_staggered = True, z_staggered = False):
	# Ensure the computational domain consists of only one grid-point in y-direction
	assert grid.ny == 1

	# Shortcuts
	u_bar, T, h, a = x_velocity_initial, temperature, mountain_height, mountain_width
	nx, nz = grid.nx, grid.nz

	# Compute Scorer parameter
	l = np.sqrt((g ** 2) / (cp * T  * (u_bar ** 2)) - (g ** 2) / (4. * (Rd ** 2) * (T ** 2)))

	# Build the underlying x-z grid
	xv = grid.x_half_levels.values if x_staggered else grid.x.values
	zv = grid.z_half_levels.values if z_staggered else grid.z.values
	x, theta = np.meshgrid(xv, zv, indexing = 'ij')
	
	# The topography
	zs = h * (a ** 2) / ((x ** 2) + (a ** 2))

	# The geometric height
	theta_s = grid.z_half_levels.values[-1]
	z = zs + cp * T / g * np.log(theta / theta_s)
	dz_dx = - 2. * h * (a ** 2) * x / (((x ** 2) + (a ** 2)) ** 2)
	dz_dtheta = cp * T / (g * theta)

	# Compute mean pressure
	p_bar = p_ref * (T / theta) ** (cp / Rd)

	# Base and mean density
	rho_ref = p_ref / (Rd * T)
	rho_bar = p_bar / (Rd * T)
	drho_bar_dtheta = - cp * p_ref / ((Rd ** 2) * (T ** 2)) * ((T / theta) ** (cp / Rd + 1.))

	# Compute the streamlines displacement and its derivative
	d = ((rho_bar / rho_ref) ** (-0.5)) * h * a * (a * np.cos(l * z) - x * np.sin(l * z)) / ((x ** 2) + (a ** 2))
	dd_dx = - ((rho_bar / rho_ref) ** (-0.5)) * h * a / (((x ** 2) + (a ** 2)) ** 2) * \
			(((a * np.sin(l * z) + x * np.cos(l * z)) * l * dz_dx + np.sin(l * z)) * ((x ** 2) + (a ** 2)) +
			 2. * x * (a * np.cos(l * z) - x * np.sin(l * z)))
	dd_dtheta = 0.5 * cp / (Rd * T) * ((theta / T) ** (0.5 * cp / Rd - 1.)) * \
				h * a * (a * np.cos(l * z) - x * np.sin(l * z)) / ((x ** 2) + (a ** 2)) - \
				((theta / T) ** (0.5 * cp / Rd)) * h * a * (a * np.sin(l * z) + x * np.cos(l * z)) * l * dz_dtheta / \
				((x ** 2) + (a ** 2))
	dd_dz = dd_dtheta / dz_dtheta

	# Compute the horizontal and vertical velocity
	u = u_bar * (1. - drho_bar_dtheta * d / (dz_dtheta * rho_bar) - dd_dz)
	w = u_bar * dd_dx

	return u, w

