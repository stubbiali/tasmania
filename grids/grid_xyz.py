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
from datetime import timedelta
import math
import numpy as np
import sympl

from tasmania.grids.topography import Topography2d
from tasmania.grids.grid_xy import GridXY
from tasmania.namelist import datatype
from tasmania.utils.utils import smaller_than as lt, \
								 smaller_or_equal_than as le


class GridXYZ:
	"""
	This class represents a rectangular and regular three-dimensional grid
	embedded in a reference system whose coordinates are

		* the first horizontal coordinate :math:`x`;
		* the second horizontal coordinate :math:`y`;
		* the vertical (terrain-following) coordinate :math:`z`. 	

	The vertical coordinate :math:`z` may be formulated to define a hybrid
	terrain-following coordinate system with terrain-following coordinate
	lines between the surface terrain-height and :math:`z = z_F`, where
	:math:`z`-coordinate lines change back to flat horizontal lines. However,
	no assumption is made on the actual nature of :math:`z`, which may be either
	pressure-based or height-based.

	Attributes
	----------
	xy_grid : grid
		:class:`~tasmania.grids.grid_xy.GridXY` representing the horizontal grid.
	z : dataarray_like
		:class:`sympl.DataArray` storing the :math:`z`-main levels.
	z_on_interface_levels : dataarray_like
		:class:`sympl.DataArray` storing the :math:`z`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`z`-spacing.
	z_interface : float
		The interface coordinate :math:`z_F`.
	topography : topography
		:class:`~tasmania.grids.topography.Topography2d` representing the
		underlying topography.
	
	Note
	----
	For the sake of compliancy with the `COSMO model <http://cosmo-model.org>`_,
	the vertical grid points are ordered from the top of the domain to the surface.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, domain_z, nz, 
				 dims_x='longitude', units_x='degrees_east', 				 
				 dims_y='latitude', units_y='degrees_north', 
				 dims_z='z', units_z='m', z_interface=None, dtype=datatype,
				 topo_type='flat_terrain', topo_time=timedelta(), topo_kwargs=None):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			The interval which the domain includes along the :math:`x`-axis.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_y : tuple
			The interval which the domain includes along the :math:`y`-axis.
		ny : int
			Number of mass points along :math:`y`.
		domain_z : tuple
			The interval which the domain includes along the :math:`z`-axis.
			This should be specified in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate. Defaults to 'longitude'.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'degrees_east'.
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate. Defaults to 'latitude'.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'degrees_north'.
		dims_z : `str`, optional
			Label for the :math:`z`-coordinate. Defaults to 'z'.
		units_z : `str`, optional
			Units for the :math:`z`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'm'.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that
			:math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the
			top of the domain. In other words, the coordinate system is supposed
			fully terrain-following.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'.
			See :class:`~tasmania.grids.topography.Topography2d` for further details.
		topo_time : `timedelta`, optional
			:class:`datetime.timedelta` representing the simulation time after
			which the topography should stop increasing. Default is 0, corresponding
			to a time-invariant terrain surface-height.
			See :class:`~tasmania.grids.topography.Topography2d` for further details.
		topo_kwargs : `dict`, optional
			Keyword arguments to be forwarded to the constructor of
			:class:`~tasmania.grids.topography.Topography2d`.

		Raises
		------
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# xy-grid
		self.xy_grid = GridXY(domain_x, nx, domain_y, ny,
							  dims_x=dims_x, units_x=units_x,
							  dims_y=dims_y, units_y=units_y)

		# z-coordinates of the half-levels
		zv_hl = np.linspace(domain_z[0], domain_z[1], nz+1, dtype=dtype)
		self.z_on_interface_levels = sympl.DataArray(zv_hl,
													 coords=[zv_hl], dims=dims_z,
													 name='z_on_interface_levels',
													 attrs={'units': units_z})

		# z-coordinates of main-levels
		zv = 0.5 * (zv_hl[:-1] + zv_hl[1:])
		self.z = sympl.DataArray(zv, coords=[zv], dims=dims_z, name='z',
								 attrs={'units': units_z})

		# Number of vertical main levels
		self.nz = int(nz)

		# z-spacing
		self.dz = math.fabs(domain_z[1] - domain_z[0]) / float(nz)

		# z-interface
		if z_interface is None:
			z_interface = domain_z[0]
		if lt(domain_z[0], domain_z[1]):
			if not (le(domain_z[0], z_interface) and le(z_interface, domain_z[1])):
				raise ValueError('z_interface should be in the range '
								 '({}, {}).'.format(domain_z[0], domain_z[1]))
		else:
			if not (le(domain_z[1], z_interface) and le(z_interface, domain_z[0])):
				raise ValueError('z_interface should be in the range '
								 '({}, {}).'.format(domain_z[1], domain_z[0]))
		self.z_interface = z_interface

		# Underlying topography
		if topo_kwargs is None or not isinstance(topo_kwargs, dict):
			self.topography = Topography2d(self.xy_grid, topo_type, topo_time)
		else:
			self.topography = Topography2d(self.xy_grid, topo_type, topo_time,
										   **topo_kwargs)

	@property
	def x(self):
		"""
		Get the :math:`x`-coordinates of the mass points.

		Return
		------
		dataarray_like :
			:class:`sympl.DataArray` storing the :math:`x`-coordinates
			of the mass points.
		"""
		return self.xy_grid.x

	@property
	def x_at_u_locations(self):
		"""
		Get the :math:`x`-coordinates of the :math:`x`-staggered points.

		Return
		------
		dataarray_like :
			:class:`sympl.DataArray` storing the :math:`x`-coordinates
			of the :math:`x`-staggered points.
		"""
		g = self.xy_grid
		return g.x_at_u_locations if hasattr(g, 'x_at_u_locations') else g.x_half_levels

	@property
	def nx(self):
		"""
		Get the number of grid points in the :math:`x`-direction.

		Return
		------
		int : 
			Number of grid points in the :math:`x`-direction.
		"""
		return self.xy_grid.nx

	@property
	def dx(self):
		"""
		Get the :math:`x`-spacing.

		Return
		------
		float : 
			The :math:`x`-spacing.
		"""
		return self.xy_grid.dx

	@property
	def y(self):
		"""
		Get the :math:`y`-coordinates of the mass points.

		Return
		------
		dataarray_like :
			:class:`sympl.DataArray` storing the :math:`y`-coordinates
			of the mass points.
		"""
		return self.xy_grid.y

	@property
	def y_at_v_locations(self):
		"""
		Get the :math:`y`-coordinates of the :math:`y`-staggered points.

		Return
		------
		dataarray_like :
			:class:`sympl.DataArray` storing the :math:`y`-coordinates
			of the :math:`y`-staggered points.
		"""
		g = self.xy_grid
		return g.y_at_v_locations if hasattr(g, 'y_at_v_locations') else g.y_half_levels

	@property
	def ny(self):
		"""
		Get the number of grid points in the :math:`y`-direction.

		Return
		------
		int : 
			Number of grid points in the :math:`y`-direction.
		"""
		return self.xy_grid.ny

	@property
	def dy(self):
		"""
		Get the :math:`y`-spacing.

		Return
		------
		float : 
			The :math:`y`-spacing.
		"""
		return self.xy_grid.dy

	@property
	def topography_height(self):
		"""
		Get the topography (i.e., terrain-surface) height.

		Return
		------
		array_like : 
			2-D :class:`numpy.ndarray` representing the topography height.
		"""
		return self.topography.topo.values

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. 

		Parameters
		----------
		time : timedelta
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self.topography.update(time)
