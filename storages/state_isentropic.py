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
from scipy.interpolate import RectBivariateSpline
import xarray as xr

from tasmania.namelist import cp, datatype, g, p_ref, Rd
from tasmania.storages.grid_data import GridData
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import convert_datetime64_to_datetime
import tasmania.utils.utils_plot as utils_plot

class StateIsentropic(GridData):
	"""
	This class inherits :class:`~storages.grid_data.GridData` to represent the state of the three-dimensional 
	(moist) isentropic model. The stored variables might be:

	* air_density (unstaggered);
	* air_isentropic_density (unstaggered);
	* x_velocity (:math:`x`-staggered);
	* y_velocity (:math:`y`-staggered);
	* x_momentum_isentropic (unstaggered);
	* y_momentum_isentropic (unstaggered);
	* air_pressure (:math:`z`-staggered);
	* exner_function (:math:`z`-staggered);
	* montgomery_potential (unstaggered);
	* height (:math:`z`-staggered);
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
		:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
	"""
	def __init__(self, time, grid, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the time instant at which the state is defined.
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.

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
		air_pressure : array_like
			:class:`numpy.ndarray` representing the pressure.
		exner_function : array_like
			:class:`numpy.ndarray` representing the Exner function.
		montgomery_potential : array_like
			:class:`numpy.ndarray` representing the Montgomery potential.
		height : array_like
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

	def contour_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate a contour plot of a field at the cross-section :math:`y = \\bar{y}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'x_velocity_unstaggered_perturbation', for the discrepancy of the :math:`x`-velocity with respect to 
				the initial condition; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;

			* 'vertical_velocity', for the vertical velocity; only for steady-state flows; the current object must contain
				the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- height.

		y_level : int 
			Index corresponding to the :math:`y`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`utils.utils_plot.contour_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		dx = self.grid.dx

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

			u = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			h = 0.5 * (self._vars['height'].values[:, y_level, :-1, time_level] +
					   self._vars['height'].values[:, y_level, 1:, time_level])
			h = 0.5 * (h[:-1, :] + h[1:, :])
			height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)

			var = u * (height[1:, :] - height[:-1, :]) / self.grid.dx
		elif field_to_plot == 'temperature': # in case temperature is not a stored variable
			z = self.grid.z_half_levels.values
			p = self._vars['pressure'].values[:, y_level, :, time_level]
			exn = self._vars['exner_function'].values[:, y_level, :, time_level]

			var = np.zeros((nx, nz + 1), dtype = datatype)
			for k in range(nz + 1):
				var[:, k] = exn[:, k] * z[k] / cp 
		else:
			raise RuntimeError('Unknown field to plot.')

		# Plot
		utils_plot.contour_xz(self.grid, self._vars['height'].values[:, y_level, :, time_level], var, **kwargs)
	
	def contourf_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate a contourf plot of a field at the cross-section :math:`y = \\bar{y}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'x_velocity_unstaggered_perturbation', for the discrepancy of the :math:`x`-velocity with respect to 
				the initial condition; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;

			* 'vertical_velocity', for the vertical velocity; only for steady-state flows; the current object must contain
				the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- height.

		y_level : int 
			Index corresponding to the :math:`y`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`utils.utils_plot.contourf_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		dx = self.grid.dx

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

			u = self._vars['x_momentum_isentropic'].values[:, y_level, :, time_level] / \
				self._vars['air_isentropic_density'].values[:, y_level, :, time_level] 
			h = 0.5 * (self._vars['height'].values[:, y_level, :-1, time_level] +
					   self._vars['height'].values[:, y_level, 1:, time_level])
			h = 0.5 * (h[:-1, :] + h[1:, :])
			height = np.concatenate((h[0:1, :], h, h[-1:, :]), axis = 0)

			var = u * (height[1:, :] - height[:-1, :]) / self.grid.dx
		elif field_to_plot == 'temperature':
			z = self.grid.z_half_levels.values
			p = self._vars['pressure'].values[:, y_level, :, time_level]
			exn = self._vars['exner_function'].values[:, y_level, :, time_level]

			var = np.zeros((nx, nz + 1), dtype = datatype)
			for k in range(nz + 1):
				var[:, k] = exn[:, k] * z[k] / cp 
		else:
			raise RuntimeError('Unknown field to plot.')

		# Plot
		utils_plot.contourf_xz(self.grid, self.grid.topography_height[:, y_level], 
							   self._vars['height'].values[:, y_level, :, time_level], var, **kwargs)

	def animation_contourf_xz(self, field_to_plot, y_level, destination, **kwargs):
		"""
		Generate a contourf animation of a field at the cross-section :math:`y = \\bar{y}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object.

		y_level : int 
			Index corresponding to the :math:`y`-level identifying the cross-section to plot.
		destination : str
			String specify the path to the location where the movie will be saved. Note that the extension should be 
			specified as well.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`utils.utils_plot.animation_contourf_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		time = self._vars[list(self._vars.keys())[0]].coords['time'].values 
		nt = len(time)
		dx = self.grid.dx

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, :] 
		elif field_to_plot == 'air_temperature':
			z = self.grid.z_half_levels.values
			p = self._vars['pressure'].values[:, y_level, :, :]
			exn = self._vars['exner_function'].values[:, y_level, :, :]

			var = np.zeros((nx, nz + 1, nt), dtype = datatype)
			for k in range(nz + 1):
				var[:, k, :] = exn[:, k, :] * z[k] / cp 
		else:
			raise RuntimeError('Unknown field to plot.')

		# Plot
		utils_plot.animation_contourf_xz(time, self.grid, self.grid.topography_height[:, y_level], 
							   			 self._vars['height'].values[:, y_level, :, :], var, destination, **kwargs)

	def contourf_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Generate a contourf plot of a field at the level :math:`\\theta = \\bar{\\theta}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'horizontal_velocity', for the horizontal velocity; the current object must contain the following variables:
			
				- air_isentropic_density;
				- x_momentum_isentropic;
				- y_momentum_isentropic.

		z_level : int 
			Index corresponding to the :math:`\\theta`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`utils.utils_plot.contourf_xy` for the complete list.
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

		# Plot
		utils_plot.contourf_xy(self.grid, var, **kwargs)

	def quiver_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Generate a quiver plot of a vector field at the level :math:`\\theta = \\bar{\\theta}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* 'horizontal_velocity', for the horizontal velocity; the current object must contain the following variables:

				- air_isentropic_density;
				- x_momentum_isentropic;
				- y_momentum_isentropic.

		z_level : int 
			Index corresponding to the :math:`\\theta`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. 
			See :func:`utils.utils_plot.quiver_xy` for the complete list.
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

		# Plot
		utils_plot.quiver_xy(self.grid, vx, vy, scalar, **kwargs)

