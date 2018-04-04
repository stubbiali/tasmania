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
Two- and three-dimensional :math:`\sigma`-terrain-following grids. 
"""
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import sys
import xarray as xr

import tasmania.namelist as nl 
from tasmania.grids.axis import Axis
from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.utils import equal_to as eq
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import smaller_or_equal_than as le

class Sigma2d(GridXZ):
	"""
	This class inherits :class:`~grids.grid_xz.GridXZ` to represent a rectangular and regular
	two-dimensional grid embedded in a reference system whose coordinates are
		* the horizontal coordinate :math:`x`; 
		* the pressure-based terrain-following coordinate :math:`\sigma = p / p_{SL}`, \
			where :math:`p` is the pressure and :math:`p_{SL}` the pressure at the sea level. 	
	The vertical coordinate :math:`\sigma` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`\sigma = \sigma_F`, where 
	:math:`\sigma`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	x : obj
		:class:`~grids.axis.Axis` representing the :math:`x`-axis.
	nx : int
		Number of grid points along :math:`x`.
	dx : float
		The :math:`x`-spacing.
	z : obj
		:class:`~grids.axis.Axis` representing the :math:`\sigma`-main levels.
	z_half_levels : obj
		:class:`~grids.axis.Axis` representing the :math:`\sigma`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`\sigma`-spacing.
	z_interface : float
		The interface coordinate :math:`\sigma_F`.
	height : obj 
		:class:`xarray.DataArray` representing the geometric height of the main levels.
	height_half_levels : obj 
		:class:`xarray.DataArray` representing the geometric height of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\sigma = \sigma_F`.
	reference_pressure : obj
		:class:`xarray.DataArray` representing the reference pressure at the main levels.
	reference_pressure_half_levels : obj
		:class:`xarray.DataArray` representing the reference pressure at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_z, nz, units_x = 'm', dims_x = 'x', z_interface = None,
				 topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{left}, ~ x_{right})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_z : tuple
			Tuple in the form :math:`(\sigma_{top}, ~ \sigma_{surface})`.
		nz : int 
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`grids.axis.Axis.__init__`).
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`\sigma_F`. If not specified, it is assumed that :math:`\sigma_F = \sigma_T`, 
			with :math:`\sigma_T` the value of :math:`\sigma` at the top of the domain. In other words, a fully 
			terrain-following coordinate system is supposed.
		topo_type : `str`, optional
			Topography type. Default is 'flat_terrain'. See :mod:`grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. See :mod:`grids.topography` 
			for further details.

		Keyword arguments
		-----------------
		**kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~grids.topography.Topography1d`.
		"""
		# Preliminary checks
		if not (le(0., domain_z[0]) or eq(domain_z[1], 1.)):
			raise ValueError('Pressure-based vertical coordinate should be positive' \
							 'and one at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_z, nz, units_x = units_x, dims_x = dims_x,
			units_z = '1', dims_z = 'atmosphere_hybrid_sigma_pressure_coordinate', # CF Conventions
			z_interface = z_interface, topo_type = topo_type, topo_time = topo_time, **kwargs)
		
		# Interface height
		self.height_interface = nl.Rd / nl.g * np.log(1 / self.z_interface) * \
				  				(nl.T_sl - 0.5 * nl.beta * np.log(1 / self.z_interface))

		# Compute geometric height and refence pressure
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. In turn, the metric terms are re-computed.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self.topography.update(time)
		self._update_metric_terms()

	def plot(self, **kwargs):
		"""
		Plot the grid half levels using :mod:`matplotlib.pyplot`'s utilities.

		Keyword arguments
		-----------------
		**kwargs : 
			Keyword arguments to be forwarded to :func:`matplotlib.pyplot.subplots`.

		Note
		----
		For the sake of compliancy with the notation employed by `COSMO <http://www.cosmo-model.org>`_,
		the vertical geometric height is denoted by :math:`z`. 
		"""
		if not kwargs:
			kwargs = {'figsize': [11,8]}
		fig, ax = plt.subplots(**kwargs)

		# Shortcuts
		x, z_hl, zf = self.x.values, self.height_half_levels.values, self.height_interface
		
		for i in range(0, z_hl.shape[1]):
			ax.plot(x, z_hl[:,i], color = 'black')
		ax.fill_between(x, 0, z_hl[:,-1], color = 'gray')

		ax.set(xlabel = '$x$ [$m$]', ylabel = '$z$ [$m$]')

		if lt(zf, z_hl[0,0]): 
			ax.text(x[-1], zf, ' $z_F$', horizontalalignment = 'left',
					verticalalignment = 'center')
			ax.plot(np.array([x[0], x[-1]]), np.array([zf, zf]),
					color = 'black', linestyle = '--')

		ax.text(x[-1], z_hl[0,0], ' $z_T$', horizontalalignment = 'left',
				verticalalignment = 'center')

		plt.show()

	def _update_metric_terms(self):
		"""
		Update the class by computing the metric terms, i.e., the geometric height and the reference pressure, 
		at both half and main levels. In doing this, a logarithmic vertical profile of reference pressure is assumed. 
		This method should be called every time the topography is updated or changed.
		"""
		# Shortcuts
		hs = self.topography.topo.values
		zv = np.reshape(self.z_half_levels.values[:,np.newaxis], (1, self.nz+1))
		zt = zv[0,0]
		zf = self.z_interface
		
		# Reference pressure at terrain surface
		if eq(nl.beta, 0.):
			p0_s = nl.p_sl * np.exp(- nl.g * hs / (nl.Rd * nl.T_sl))
		else:
			p0_s = nl.p_sl * np.exp(- nl.T_sl / nl.beta * \
				   (1. - np.sqrt(1. - 2. * nl.beta * nl.g * hs / (nl.Rd * nl.T_sl * nl.T_sl))))
		p0_s = np.repeat(p0_s[:,np.newaxis], self.nz+1, axis = 1)

		# Reference pressure at half levels
		a = nl.p_sl * zv * (np.logical_and(le(zt, zv), le(zv, zf))) + \
			nl.p_sl * zf * (1. - zv) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		a = np.repeat(a, self.nx, axis = 0)
		b = (zv - zf) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		b = np.repeat(b, self.nx, axis = 0)
		p0_hl = a + b * p0_s
		self.reference_pressure_half_levels = xr.DataArray \
			(p0_hl, coords = [self.x.values, self.z_half_levels.values],
			 dims = [self.x.dims, self.z_half_levels.dims], attrs = {'units': 'Pa'})

		# Half levels geometric height
		if eq(nl.beta, 0.):
			z_hl = nl.Rd * nl.T_sl / nl.g * np.log(nl.p_sl / p0_hl)
		else:
			z_hl = nl.Rd / nl.g * np.log(nl.p_sl / p0_hl) * \
				   (nl.T_sl - 0.5 * nl.beta * np.log(nl.p_sl / p0_hl))
		self.height_half_levels = xr.DataArray(z_hl, coords = [self.x.values, self.z_half_levels.values],
											   dims = [self.x.dims, self.z_half_levels.dims], 
											   attrs = {'units': 'm'})

		# Reference pressure at main levels
		self.reference_pressure = xr.DataArray(0.5 * (p0_hl[:,:-1] + p0_hl[:,1:]), 
											   coords = [self.x.values, self.z.values],
											   dims = [self.x.dims, self.z.dims], attrs = {'units': 'Pa'})
		# Main levels geometric height
		self.height = xr.DataArray(0.5 * (z_hl[:,:-1] + z_hl[:,1:]),
								   coords = [self.x.values, self.z.values],
								   dims = [self.x.dims, self.z.dims], attrs = {'units': 'm'})

class Sigma3d(GridXYZ):
	"""
	This class inherits :class:`~grids.grid_xyz.GridXYZ` to represent a rectangular and regular computational grid 
	embedded in a three-dimensional terrain-following reference system, whose coordinates are:
		* first horizontal coordinate :math:`x`, e.g., the longitude;
		* second horizontal coordinate :math:`y`, e.g., the latitude;
		* the pressure-based terrain-following coordinate :math:`\sigma = p / p_{SL}`, \
			where :math:`p` is the pressure and :math:`p_{SL}` the pressure at the sea level. 	
	The vertical coordinate :math:`\sigma` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`\sigma = \sigma_F`, where 
	:math:`\sigma`-coordinate lines change back to flat horizontal lines. 

	Attributes
	----------
	xy_grid : obj
		:class:`~grids.grid_xy.GridXY` representing the horizontal grid.
	z : obj
		:class:`~grids.axis.Axis` representing the :math:`\sigma`-main levels.
	z_half_levels : obj
		:class:`~grids.axis.Axis` representing the :math:`\sigma`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`\sigma`-spacing.
	z_interface : float
		The interface coordinate :math:`\sigma_F`.
	topography : obj
		:class:`~tasmania.grids.topography.Topography2d` representing the underlying topography.
	height : obj
		:class:`xarray.DataArray` representing the geometric height of the main levels.
	height_half_levels : obj
		:class:`xarray.DataArray` representing the geometric height of the half levels.
	height_interface : float
		Geometric height corresponding to :math:`\sigma = \sigma_F`.
	reference_pressure : obj
		:class:`xarray.DataArray` storing the reference pressure at the main levels.
	reference_pressure_half_levels : obj
		:class:`xarray.DataArray` storing the reference pressure at the half levels.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, domain_z, nz, 
				 units_x = 'degrees_east', dims_x = 'longitude', units_y = 'degrees_north', dims_y = 'latitude', 
				 z_interface = None, topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{left}, ~ x_{right})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_y : tuple
			Tuple in the form :math:`(y_{left}, ~ y_{right})`.
		ny : int
			Number of grid points in the :math:`y`-direction.
		domain_z : tuple
			Tuple in the form :math:`(\sigma_{top}, ~ \sigma_{surface})`.
		nz : int
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`grids.axis.Axis.__init__`).
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`grids.axis.Axis.__init__`).
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`\sigma_F`. If not specified, it is assumed that :math:`\sigma_F = \sigma_T`, 
			with :math:`\sigma_T` the value of  :math:`\sigma` at the top of the domain. In other words, a fully 
			terrain-following coordinate system is supposed.
		topo_type : `str`, optional
			Topography type. Default is 'flat_terrain'. See :mod:`grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. 
			See :mod:`grids.topography` for further details.

		Keyword arguments
		-----------------
		**kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~grids.topography.Topography2d`.
		"""
		# Preliminary checks
		if not (le(0., domain_z[0]) or eq(domain_z[1], 1.)):
			raise ValueError('Pressure-based vertical coordinate should be positive' \
							 'and one at the terrain surface.')

		# Call parent's constructor
		super().__init__(domain_x, nx, domain_y, ny, domain_z, nz,
			units_x = units_x, dims_x = dims_x, units_y = units_y, dims_y = dims_y,
			units_z = '1', dims_z = 'atmosphere_hybrid_sigma_pressure_coordinate', # CF Conventions
			z_interface = z_interface, topo_type = topo_type, topo_time = topo_time, **kwargs)
		
		# Interface height
		self.height_interface = nl.Rd / nl.g * np.log(1 / self.z_interface) * \
				  				(nl.T_sl - 0.5 * nl.beta * np.log(1 / self.z_interface))

		# Compute geometric height and refence pressure
		self._update_metric_terms()

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. In turn, the metric terms are re-computed.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self.topography.update(time)
		self._update_metric_terms()

	def _update_metric_terms(self):
		"""
		Update the class by computing the metric terms, i.e., the geometric height and the reference pressure, 
		at both half and main levels. In doing this, a logarithmic vertical profile of reference pressure is assumed. 
		This method should be called every time the topography is updated or changed.
		"""
		# Shortcuts
		hs = np.repeat(self.topography.topo.values[:,:,np.newaxis], self.nz+1, axis = 2)
		zv = np.reshape(self.z_half_levels.values[:,np.newaxis,np.newaxis], (1, 1, self.nz+1))
		zt = zv[0,0]
		zf = self.z_interface

		# Reference pressure at terrain surface
		if eq(nl.beta, 0.):
			p0_s = nl.p_sl * np.exp(- nl.g * hs / (nl.Rd * nl.T_sl))
		else:
			p0_s = nl.p_sl * np.exp(- nl.T_sl / nl.beta * \
				   (1. - np.sqrt(1. - 2. * nl.beta * nl.g * hs / (nl.Rd * nl.T_sl * nl.T_sl))))

		# Reference pressure at half levels.
		a = nl.p_sl * zv * (np.logical_and(le(zt, zv), le(zv, zf))) + \
			nl.p_sl * zf * (1. - zv) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		a = np.tile(a, (self.nx, self.ny, 1))
		b = (zv - zf) / (1. - zf) * (np.logical_and(lt(zf, zv), le(zv, 1.)))
		b = np.tile(b, (self.nx, self.ny, 1))
		p0_hl = a + b * p0_s
		self.reference_pressure_half_levels = xr.DataArray(p0_hl,
			coords = [self.x.values, self.y.values, self.z_half_levels.values],
			dims = [self.x.dims, self.y.dims, self.z_half_levels.dims], attrs = {'units': 'Pa'})

		# Half levels geometric height
		if eq(nl.beta, 0.):
			z_hl = nl.Rd * nl.T_sl / nl.g * np.log(nl.p_sl / p0_hl)
		else:
			z_hl = nl.Rd / nl.g * np.log(nl.p_sl / p0_hl) * \
				   (nl.T_sl - 0.5 * nl.beta * np.log(nl.p_sl / p0_hl))
		self.height_half_levels = xr.DataArray(z_hl,
			coords = [self.x.values, self.y.values, self.z_half_levels.values],
			dims = [self.x.dims, self.y.dims, self.z_half_levels.dims], attrs = {'units': 'm'})

		# Reference pressure at main levels
		self.reference_pressure = xr.DataArray(0.5 * (p0_hl[:,:,:-1] + p0_hl[:,:,1:]), 
			coords = [self.x.values, self.y.values, self.z.values],
			dims = [self.x.dims, self.y.dims, self.z.dims], attrs = {'units': 'Pa'})

		# Main levels geometric height
		self.height = xr.DataArray(0.5 * (z_hl[:,:,:-1] + z_hl[:,:,1:]),
			coords = [self.x.values, self.y.values, self.z.values],
			dims = [self.x.dims, self.y.dims, self.z.dims], attrs = {'units': 'm'})

