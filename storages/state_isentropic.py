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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, RectBivariateSpline
import xarray as xr

from tasmania.namelist import cp, datatype, g, p_ref, Rd
from tasmania.storages.grid_data import GridData
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import convert_datetime64_to_datetime
import tasmania.utils.utils_plot as utils_plot

class StateIsentropic(GridData):
	"""
	This class inherits :class:`~tasmania.storages.grid_data.GridData` to represent the state of the three-dimensional 
	(moist) isentropic model. The stored variables might be:

	* air_density (unstaggered);
	* air_isentropic_density (unstaggered);
	* x_velocity (:math:`x`-staggered);
	* y_velocity (:math:`y`-staggered);
	* x_momentum_isentropic (unstaggered);
	* y_momentum_isentropic (unstaggered);
	* air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
	* exner_function or exner_function_on_interface_levels (:math:`z`-staggered);
	* montgomery_potential (unstaggered);
	* height or height_on_interface_levels (:math:`z`-staggered);
	* air_temperature (unstaggered);
	* mass_fraction_water_vapor_in_air (unstaggered);
	* water_vapor_isentropic_density (unstaggered);
	* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
	* cloud_liquid_water_isentropic_density (unstaggered);
	* mass_fraction_of_precipitation_water_in_air (unstaggered);
	* precipitation_water_isentropic_density (unstaggered).

	Note
	----
	At any point, an instance of this class may or may not contain all the listed model variables. Indeed, variables
	might be added gradually, according to user's needs.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
	"""
	def __init__(self, time, grid, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the time instant at which the state is defined.
		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.

		Keyword arguments
		-----------------
		air_density : array_like
			:class:`numpy.ndarray` representing the density.
		air_isentropic_density : array_like
			:class:`numpy.ndarray` representing the isentropic density.
		x_velocity : array_like
			:class:`numpy.ndarray` representing the :math:`x`-velocity.
		y_velocity : array_like
			:class:`numpy.ndarray` representing the :math:`y`-velocity.
		x_momentum_isentropic : array_like
			:class:`numpy.ndarray` representing the (isentropic) :math:`x`-momentum.
		y_momentum_isentropic : array_like
			:class:`numpy.ndarray` representing the (isentropic) :math:`y`-momentum.
		air_pressure or air_pressure_on_interface_levels : array_like
			:class:`numpy.ndarray` representing the pressure.
		exner_function or exner_function_on_interface_levels: array_like
			:class:`numpy.ndarray` representing the Exner function.
		montgomery_potential : array_like
			:class:`numpy.ndarray` representing the Montgomery potential.
		height or height_on_interface_levels: array_like
			:class:`numpy.ndarray` representing the geometrical height of the half-levels.
		air_temperature : array_like
			:class:`numpy.ndarray` representing the temperature.
		mass_fraction_of_water_vapor_in_air : array_like
			:class:`numpy.ndarray` representing the mass fraction of water vapor.
		water_vapor_isentropic_density : array_like
			:class:`numpy.ndarray` representing the isentropic density of water vapor.
		mass_fraction_of_cloud_liquid_water_in_air : array_like
			:class:`numpy.ndarray` representing the mass fraction of cloud water.
		cloud_liquid_water_isentropic_density : array_like
			:class:`numpy.ndarray` representing the isentropic density of cloud water.
		mass_fraction_of_precipitation_water_in_air : array_like
			:class:`numpy.ndarray` representing the mass fraction of precipitation water.
		precipitation_water_isentropic_density : array_like
			:class:`numpy.ndarray` representing the isentropic density of precipitation water.
		"""
		super().__init__(time, grid, **kwargs)

	def get_cfl(self, dt):
		"""
		Compute the CFL number. 

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.

		Return
		------
		float :
			The CFL number.

		Note
		----
		If the CFL number exceeds the unity, i.e., if the CFL condition is violated, the method throws a warning.
		"""
		u_max = np.amax(np.absolute(self._vars['x_velocity'].values[:,:,:,-1]))
		v_max = np.amax(np.absolute(self._vars['y_velocity'].values[:,:,:,-1]))

		cfl_x = u_max * (1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds) / self.grid.dx
		cfl_y = v_max * (1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds) / self.grid.dy
		cfl = max(cfl_x, cfl_y)

		if cfl > 1.:
			raise RuntimeWarning('CFL condition violated.')
		elif np.isnan(cfl):
			raise RuntimeWarning('NaN values.')

		return cfl

	#
	# Plotting utilities
	#
	def contour_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate the contour plot of a field at a cross-section parallel to the :math:`xz`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'x_velocity_unstaggered_perturbation', for the discrepancy of the :math:`x`-velocity with respect to \
				the initial condition; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;

			* 'vertical_velocity', for the vertical velocity; only for two-dimensional steady-state flows; \
				the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- height or height_on_interface_levels.

		y_level : int 
			:math:`y`-index identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.contour_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, time_level] 
		elif field_to_plot == 'x_velocity_unstaggered_perturbation':
			u_start = self._vars['x_momentum_isentropic'].values[:, y_level, :, 0] / \
					  self._vars['air_isentropic_density'].values[:, y_level, :, 0] 
			u_final = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
					  self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 

			var = u_final - u_start
		elif field_to_plot == 'vertical_velocity':
			assert self.grid.ny == 1

			u      = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				   	 self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			h_da   = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
			h      = 0.5 * (h_da.values[:, y_level, :-1, time_level] + h_da.values[:, y_level, 1:, time_level])
			h      = 0.5 * (h[:-1, :] + h[1:, :])
			height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)

			var    = u * (height[1:, :] - height[:-1, :]) / self.grid.dx
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		ni, nk = var.shape

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nk, axis = 1)

		# The underlying z-grid
		z_da = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
		z_   = np.copy(z_da.values[:, y_level, :, time_level])
		z    = z_ if nk == nz + 1 else 0.5 * (z_[:,:-1] + z_[:,1:])
		zv   = np.zeros((ni, nk), dtype = datatype)
		if ni == nx:
			zv[:, :] = z[:, :]
		else:
			zv[1:-1, :] = 0.5 * (z[:-1, :] + z[1:, :])
			zv[0, :], zv[-1, :] = zv[1, :], zv[-2, :]

		# The underlying topography
		topography_ = z_da.values[:, y_level, -1, time_level]
		if ni == nx:
			topography = topography_
		else:
			topography = np.zeros((nx + 1), dtype = datatype)
			topography[1:-1] = 0.5 * (topography_[:-1] + topography_[1:])
			topography[0], topography[-1] = topography[1], topography[-2]

		# Plot
		utils_plot.contour_xz(xv, zv, var, topography, **kwargs)

	def contourf_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Generate the contourf plot of a field at a cross-section parallel to the :math:`xy`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'horizontal_velocity', for the horizontal velocity; the current object must contain the \
				following variables:
			
				- air_isentropic_density;
				- x_momentum_isentropic;
				- y_momentum_isentropic.

		z_level : int 
			:math:`z`-index identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.contourf_xy` for the complete list.
		"""
		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, :, z_level, time_level]
		elif field_to_plot == 'horizontal_velocity':
			u = self._vars['x_momentum_isentropic'].values[:, :, z_level, time_level] / \
				self._vars['air_isentropic_density'].values[:, :, z_level, time_level] 
			v = self._vars['y_momentum_isentropic'].values[:, :, z_level, time_level] / \
				self._vars['air_isentropic_density'].values[:, :, z_level, time_level] 
			var = np.sqrt(u ** 2 + v ** 2)
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		nx, ny = self.grid.nx, self.grid.ny
		ni, nj = var.shape

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nj, axis = 1)

		# The underlying y-grid
		y  = self.grid.y.values[:] if nj == ny else self.grid.y_at_v_locations.values[:]
		yv = np.repeat(y[np.newaxis, :], ni, axis = 0)

		# The topography height
		topography_ = np.copy(self.grid.topography_height)
		topography  = np.zeros((ni, nj), dtype = datatype) 
		if ni == nx and nj == ny:
			topography[:,:] = topography_[:,:]
		elif ni == nx + 1 and nj == ny:
			topography[1:-1,:] = 0.5 * (topography_[:-1,:] + topography_[1:,:])
			topography[0,:], topography[-1,:] = topography[1,:], topography[-2,:]
		elif ni == nx and nj == ny + 1:
			topography[:,1:-1] = 0.5 * (topography_[:,:-1] + topography_[:,1:])
			topography[:,0], topography[:,-1] = topography[:,1], topography[:,-2]
		else:
			topography[1:-1,1:-1] = 0.25 * (topography_[:-1,:-1] + topography_[1:,:-1] +
											topography_[:-1,:1]  + topography_[1:,1:])
			topography[0,1:-1], topography[-1,1:-1] = topography[1,1:-1], topography[-2,1:-1]
			topography[:,0], topography[:,-1] = topography[:,1], topography[:,-2]

		# Plot
		utils_plot.contourf_xy(xv, yv, topography, var, **kwargs)

	def contourf_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate the contourf plot of a field at a cross-section parallel to the :math:`xz`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'x_velocity_unstaggered_perturbation', for the discrepancy of the :math:`x`-velocity with respect to \
				the initial condition; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;

			* 'vertical_velocity', for the vertical velocity; only for two-dimensional steady-state flows; \
				the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- height or height_on_interface_levels.

		y_level : int 
			:math:`y`-index identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.contourf_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, time_level] 
		elif field_to_plot == 'x_velocity_unstaggered_perturbation':
			u_start = self._vars['x_momentum_isentropic'].values[:, y_level, :, 0] / \
					  self._vars['air_isentropic_density'].values[:, y_level, :, 0] 
			u_final = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
					  self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 

			var = u_final - u_start
		elif field_to_plot == 'vertical_velocity':
			assert self.grid.ny == 1

			u      = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				     self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			h_da   = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
			h      = 0.5 * (h_da.values[:, y_level, :-1, time_level] + h_da.values[:, y_level, 1:, time_level])
			h 	   = 0.5 * (h[:-1, :] + h[1:, :])
			height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)

			var = u * (height[1:, :] - height[:-1, :]) / self.grid.dx
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		ni, nk = var.shape

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nk, axis = 1)

		# The underlying z-grid
		h_da = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
		z_   = np.copy(h_da.values[:, y_level, :, time_level])
		z    = z_ if nk == nz + 1 else 0.5 * (z_[:,:-1] + z_[:,1:])
		zv = np.zeros((ni, nk), dtype = datatype)
		if ni == nx:
			zv[:, :] = z[:, :]
		else:
			zv[1:-1, :] = 0.5 * (z[:-1, :] + z[1:, :])
			zv[0, :], zv[-1, :] = zv[1, :], zv[-2, :]

		# The underlying topography
		topography_ = h_da.values[:, y_level, -1, time_level]
		if ni == nx:
			topography = topography_
		else:
			topography = np.zeros((nx + 1), dtype = datatype)
			topography[1:-1] = 0.5 * (topography_[:-1] + topography_[1:])
			topography[0], topography[-1] = topography[1], topography[-2]

		# Plot
		utils_plot.contourf_xz(xv, zv, var, topography, **kwargs)

	def quiver_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Generate the quiver plot of a vector field at a cross section parallel to the :math:`xy`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* 'horizontal_velocity', for the horizontal velocity; the current object must contain the \
				following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- y_momentum_isentropic.

		z_level : int 
			:math:`z`-level identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.quiver_xy` for the complete list.
		"""
		# Extract, compute, or interpolate the field to plot
		if field_to_plot == 'horizontal_velocity':
			vx = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				 self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			vy = self._vars['y_momentum_isentropic'].values[:, y_level, :, time_level] / \
				 self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			scalar = np.sqrt(vx ** 2 + vy ** 2)
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		nx, ny = self.grid.nx, self.grid.ny
		ni, nj = scalar.shape

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nj, axis = 1)

		# The underlying y-grid
		y  = self.grid.y.values[:] if nj == ny else self.grid.y_at_v_locations.values[:]
		yv = np.repeat(y[np.newaxis, :], ni, axis = 0)

		# The topography height
		topography_ = np.copy(self.grid.topography_height)
		topography  = np.zeros((ni, nj), dtype = datatype) 
		if ni == nx and nj == ny:
			topography[:,:] = topography_[:,:]
		elif ni == nx + 1 and nj == ny:
			topography[1:-1,:] = 0.5 * (topography_[:-1,:] + topography_[1:,:])
			topography[0,:], topography[-1,:] = topography[1,:], topography[-2,:]
		elif ni == nx and nj == ny + 1:
			topography[:,1:-1] = 0.5 * (topography_[:,:-1] + topography_[:,1:])
			topography[:,0], topography[:,-1] = topography[:,1], topography[:,-2]
		else:
			topography[1:-1,1:-1] = 0.25 * (topography_[:-1,:-1] + topography_[1:,:-1] +
											topography_[:-1,:1]  + topography_[1:,1:])
			topography[0,1:-1], topography[-1,1:-1] = topography[1,1:-1], topography[-2,1:-1]
			topography[:,0], topography[:,-1] = topography[:,1], topography[:,-2]

		# Plot
		utils_plot.quiver_xy(xv, yv, topography, vx, vy, scalar, **kwargs)

	def quiver_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate the quiver plot of a vector field at a cross section parallel to the :math:`xz`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* 'velocity', for the velocity field; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- height or height_on_interface_levels.

		y_level : int 
			:math:`y`-level identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.quiver_xz` for the complete list.
		"""
		# Extract, compute, or interpolate the field to plot
		if field_to_plot == 'velocity':
			# Extract the x-velocity
			vx = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				 self._vars['air_isentropic_density'].values[:, y_level, :, time_level]

			# Compute the (Cartesian) vertical velocity
			z_da   = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
			z      = 0.5 * (z_da.values[:, y_level, :-1, time_level] + z_da.values[:, y_level, 1:, time_level])
			h      = 0.5 * (z[:-1, :] + z[1:, :])
			height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)
			vz     = vx * (height[1:, :] - height[:-1, :]) / self.grid.dx

			# Compute the velocity magnitude
			scalar = np.sqrt(vx ** 2 + vz ** 2)
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		ni, nk = scalar.shape

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nk, axis = 1)

		# The underlying z-grid
		zv = self._vars['height'].values[:, y_level, :, time_level]
		if ni == nx + 1:
			zv_ = 0.5 * (zv[:-1,:] + zv[1:,:])
			zv  = np.concatenate((zv_[0:1,:], zv_, zv_[-1:,:]), axis = 0)
		if nk == nz:
			zv = 0.5 * (zv[:,:-1] + zv[:,1:])

		# The topography height
		topography = self.grid.topography_height[:, y_level]
		if ni == nx + 1:
			tp_ = 0.5 * (topography[:-1] + topography[1:])
			topography  = np.concatenate((tp_[:1], tp_, tp_[-1:]), axis = 0)

		# Plot
		utils_plot.quiver_xz(xv, zv, topography, vx, vz, scalar, **kwargs)

	def streamplot_xz(self, y_level, time_level, **kwargs):
		"""
		Generate the streamplot of a two-dimensional velocity field.

		Note
		----
		The current object should contain the following variables:

			* air_isentropic_density;
			* x_momentum_isentropic;
			* height or height_on_interface_levels.

		Parameters
		----------
		y_level : int 
			:math:`y`-index identifying the cross-section.
		time_level : int 
			The time level.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.streamplot_xz` for the complete list.
		"""
		# Make sure the state is two-dimensional
		assert self.grid.ny == 1

		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz

		# Extract the horizontal velocity
		u = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
			self._vars['air_isentropic_density'].values[:, y_level, :, time_level]

		# Compute the (Cartesian) vertical velocity
		z_da   = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
		z      = 0.5 * (z_da.values[:, y_level, :-1, time_level] + z_da.values[:, y_level, 1:, time_level])
		h      = 0.5 * (z[:-1, :] + z[1:, :])
		height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)
		w      = u * (height[1:, :] - height[:-1, :]) / self.grid.dx

		# Interpolation points
		x  = np.repeat(self.grid.x.values[:, np.newaxis], nz, axis = 1)
		xi = x.T.ravel()
		zi = z.T.ravel()
		pi = np.concatenate((xi[:, np.newaxis], zi[:, np.newaxis]), axis = 1)

		# Interpolation values
		ui = u.T.ravel() 
		wi = w.T.ravel()

		# Evaluation points
		m = 1000
		hb, ht = min(z[:,-1]), min(z[:,0])
		xv_ = self.grid.x.values
		zv_ = np.linspace(hb, ht, m)[::-1]
		xv, zv = np.meshgrid(xv_, zv_)
		pv = np.concatenate((xv.T.ravel()[:, np.newaxis], zv.T.ravel()[:, np.newaxis]), axis = 1)

		# Interpolate
		U = np.reshape(griddata(pi, ui, pv), (nx, m))
		W = np.reshape(griddata(pi, wi, pv), (nx, m))

		# Velocity magnitude
		color = np.sqrt(U ** 2 + W ** 2)

		# The underlying topography
		topography = z_da.values[:, y_level, -1, time_level]

		# The evaluation points below the topography are set to nan
		for i in range(nx):
			for k in range(m):
				if zv[k,i] < topography[i]:
					U[i,k]     = None
					W[i,k]     = None
					color[i,k] = None

		# Plot
		utils_plot.streamplot_xz(xv_, zv_, U.T, W.T, color.T, topography, **kwargs)

	#
	# Animation utilities
	#
	def animation_contourf_xz(self, field_to_plot, y_level, destination, **kwargs):
		"""
		Generate an animation showing the time evolution of the contourfs of a field at a cross-section 
		parallel to the :math:`xz`-plane.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object.

		y_level : int 
			:math:`y`-index identifying the cross-section.
		destination : str
			String specifying the path to the location where the movie will be saved. 
			Note that the string should include the extension as well.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`~tasmania.utils.utils_plot.animation_contourf_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, :] 
		else:
			raise RuntimeError('Unknown field to plot.')

		# Shortcuts
		ni, nk, nt = var.shape

		# Time axis
		time = self._vars[list(self._vars.keys())[0]].coords['time'].values 

		# The underlying x-grid
		x  = self.grid.x.values[:] if ni == nx else self.grid.x_at_u_locations.values[:]
		xv = np.repeat(x[:, np.newaxis], nk, axis = 1)

		# The underlying z-grid
		z_da = self._vars['height'] if self._vars['height'] is not None else self._vars['height_on_interface_levels']
		z_   = np.copy(z_da.values[:, y_level, :, :])
		z    = z_ if nk == nz + 1 else 0.5 * (z_[:, :-1, :] + z_[:, 1:, :])
		zv   = np.zeros((ni, nk, nt), dtype = datatype)
		if ni == nx:
			zv[:, :, :] = z[:, :, :]
		else:
			zv[1:-1, :, :] = 0.5 * (z[:-1, :, :] + z[1:, :, :])
			zv[0, :, :], zv[-1, :, :] = zv[1, :, :], zv[-2, :, :]

		# The underlying topography
		topography_ = z_da.values[:, y_level, -1, :]
		if ni == nx:
			topography = topography_
		else:
			topography = np.zeros((nx + 1, nt), dtype = datatype)
			topography[1:-1,:] = 0.5 * (topography_[:-1,:] + topography_[1:,:])
			topography[0,:], topography[-1,:] = topography[1,:], topography[-2,:]

		# Plot
		utils_plot.animation_contourf_xz(destination, time, xv, zv, var, topography, **kwargs)

