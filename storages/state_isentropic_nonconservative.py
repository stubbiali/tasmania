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
import numpy as np
from scipy.interpolate import RectBivariateSpline
import xarray as xr

from namelist import cp, datatype, g, p_ref, Rd
from storages.grid_data import GridData
from utils.utils import smaller_than as lt
from utils.utils import convert_datetime64_to_datetime
import utils.utils_plot as utils_plot

class StateIsentropicNonconservative(GridData):
	"""
	This class inherits :class:`~storages.grid_data.GridData` to represent the nonconservative state of the three-dimensional 
	(moist) isentropic model. The stored variables are:

	* isentropic_density (unstaggered);
	* x_velocity (:math:`x`-staggered);
	* x_velocity_unstaggered (unstaggered);
	* y_velocity (:math:`y`-staggered);
	* y_velocity_unstaggered (unstaggered);
	* pressure (:math:`z`-staggered);
	* exner_function (:math:`z`-staggered);
	* montgomery_potential (unstaggered);
	* height (:math:`z`-staggered).
	* water_vapor_mass_fraction (unstaggered, optional);
	* cloud_water_mass_fraction (unstaggered, optional);
	* precipitation_water_mass_fraction (unstaggered, optional).

	Attributes
	----------
	grid : obj
		:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
	"""
	def __init__(self, time, grid, isentropic_density, x_velocity, x_velocity_unstaggered, y_velocity, y_velocity_unstaggered,
				 pressure, exner_function, montgomery_potential, height,
				 water_vapor_mass_fraction = None, cloud_water_mass_fraction = None, precipitation_water_mass_fraction = None):
		"""
		Constructor.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the time instant at which the state is defined.
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		isentropic_density : array_like
			:class:`numpy.ndarray` representing the isentropic density.
		x_velocity : array_like
			:class:`numpy.ndarray` representing the :math:`x`-velocity.
		x_velocity_unstaggered : array_like
			:class:`numpy.ndarray` representing the unstaggered :math:`x`-velocity.
		y_velocity : array_like
			:class:`numpy.ndarray` representing the :math:`y`-velocity.
		y_velocity_unstaggered : array_like
			:class:`numpy.ndarray` representing the unstaggered :math:`y`-velocity.
		pressure : array_like
			:class:`numpy.ndarray` representing the pressure.
		exner_function : array_like
			:class:`numpy.ndarray` representing the Exner function.
		montgomery_potential : array_like
			:class:`numpy.ndarray` representing the Montgomery potential.
		height : array_like
			:class:`numpy.ndarray` representing the geometrical height of the half-levels.
		water_vapor_mass_fraction : `array_like`, optional
			:class:`numpy.ndarray` representing the mass fraction of water vapor.
		cloud_water_mass_fraction : `array_like`, optional
			:class:`numpy.ndarray` representing the mass fraction of cloud water.
		precipitation_water_mass_fraction : `array_like`, optional
			:class:`numpy.ndarray` representing the mass fraction of precipitation water.
		"""
		kwargs = {'isentropic_density'     : isentropic_density,
				  'x_velocity'             : x_velocity,
				  'x_velocity_unstaggered' : x_velocity_unstaggered,
				  'y_velocity'             : y_velocity,
				  'y_velocity_unstaggered' : y_velocity_unstaggered,
				  'pressure'               : pressure,
				  'exner_function'         : exner_function,
				  'montgomery_potential'   : montgomery_potential,
				  'height'                 : height}
		if water_vapor_mass_fraction is not None:
			kwargs['water_vapor_mass_fraction'] = water_vapor_mass_fraction
		if cloud_water_mass_fraction is not None:
			kwargs['cloud_water_mass_fraction'] = cloud_water_mass_fraction
		if precipitation_water_mass_fraction is not None:
			kwargs['precipitation_water_mass_fraction'] = precipitation_water_mass_fraction

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
			* 'x_velocity_perturbation', for the discrepancy of the :math:`x`-velocity with respect to the initial condition;
			* 'vertical_velocity', for the vertical velocity; only for steady-state flows;
			* 'temperature', for the temperature.

		y_level : int 
			Index corresponding to the :math:`y`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. See :func:`utils.utils_plot.contour_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		dx = self.grid.dx

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, time_level] 
		elif field_to_plot == 'x_velocity_perturbation':
			var = self._vars['x_velocity_unstaggered'].values[:, y_level, :, time_level] - \
				  self._vars['x_velocity_unstaggered'].values[:, y_level, :, 0]
		elif field_to_plot == 'vertical_velocity':
			assert self.grid.ny == 1

			u = self._vars['x_velocity_unstaggered'].values[:, y_level, :, time_level] 
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
		utils_plot.contour_xz(self.grid, self._vars['height'].values[:, y_level, :, time_level], var, **kwargs)
	
	def contourf_xz(self, field_to_plot, y_level, time_level, **kwargs):
		"""
		Generate a contourf plot of a field at the cross-section :math:`y = \\bar{y}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object;
			* 'x_velocity_perturbation', for the discrepancy of the :math:`x`-velocity with respect to the initial condition;
			* 'vertical_velocity', for the vertical velocity; only for steady-state flows;
			* 'temperature', for the temperature.

		y_level : int 
			Index corresponding to the :math:`y`-level identifying the cross-section to plot.
		time_level : int 
			The index corresponding to the time level to plot.
		**kwargs :
			Keyword arguments to specify different plotting settings. See :func:`utils.utils_plot.contour_xz` for the complete list.
		"""
		# Shortcuts
		nx, nz = self.grid.nx, self.grid.nz
		dx = self.grid.dx

		# Extract, compute, or interpolate the field to plot
		if field_to_plot in self._vars:
			var = self._vars[field_to_plot].values[:, y_level, :, time_level] 
		elif field_to_plot == 'x_velocity_perturbation':
			var = self._vars['x_velocity_unstaggered'].values[:, y_level, :, time_level] - \
				  self._vars['x_velocity_unstaggered'].values[:, y_level, :, 0]
		elif field_to_plot == 'vertical_velocity':
			assert self.grid.ny == 1

			u = self._vars['x_velocity_unstaggered'].values[:, y_level, :, time_level] 
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

	def contourf_xy(self, field_to_plot, z_level, time_level, **kwargs):
		"""
		Generate a contourf plot of a field at the level :math:`\\theta = \\bar{\\theta}`.

		Parameters
		----------
		field_to_plot : str 
			String specifying the field to plot. This might be:

			* the name of a variable stored in the current object.
			* 'horizontal_velocity', for the horizontal velocity.

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
			u = self._vars['x_velocity_unstaggered'].values[:, :, z_level, time_level]
			v = self._vars['y_velocity_unstaggered'].values[:, :, z_level, time_level]
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

			* 'horizontal_velocity', for the horizontal velocity.

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
			vx = self._vars['x_velocity_unstaggered'].values[:, :, z_level, time_level]
			vy = self._vars['y_velocity_unstaggered'].values[:, :, z_level, time_level]
			scalar = np.sqrt(vx ** 2 + vy ** 2)
		else:
			raise RuntimeError('Unknown field to plot.')

		# Plot
		utils_plot.quiver_xy(self.grid, vx, vy, scalar, **kwargs)

