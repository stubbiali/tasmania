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
import matplotlib.pyplot as plt
import numpy as np

from tasmania.grids.axis import Axis
from tasmania.grids.topography import Topography2d
from tasmania.grids.grid_xy import GridXY
from tasmania.namelist import datatype
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import smaller_or_equal_than as le

class GridXYZ:
	"""
	Rectangular and regular three-dimensional grid embedded in a reference system whose coordinates are

		* the first horizontal coordinate :math:`x`;
		* the second horizontal coordinate :math:`y`;
		* the vertical (terrain-following) coordinate :math:`z`. 	

	The vertical coordinate :math:`z` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`z = z_F`, where 
	:math:`z`-coordinate lines change back to flat horizontal lines. However, no assumption is made on the actual 
	nature of :math:`z` which may be either pressure-based or height-based.

	Attributes
	----------
	xy_grid : obj
		:class:`~tasmania.grids.grid_xy.GridXY` representing the horizontal grid..
	z : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`z`-main levels. 
	z_on_interface_levels : obj
		:class:`~tasmania.grids.axis.Axis` representing the :math:`z`-half levels. 
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`z`-spacing.
	z_interface : float
		The interface coordinate :math:`z_F`.
	topography : obj
		:class:`~tasmania.grids.topography.Topography2d` representing the underlying topography.
	
	Note
	----
	For the sake of compliancy with the `COSMO model <http://cosmo-model.org>`_, the vertical grid points are ordered 
	from the top of the domain to the surface.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, domain_z, nz, 
				 units_x = 'degrees_east', dims_x = 'longitude',
				 units_y = 'degrees_north', dims_y = 'latitude',
				 units_z = 'm', dims_z = 'z',
				 z_interface = None,
				 topo_type = 'flat_terrain', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{start}, ~ x_{stop})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_y : tuple
			Tuple in the form :math:`(y_{start}, ~ y_{stop})`.
		ny : int
			Number of grid points in the :math:`y`-direction.
		domain_z : tuple
			Tuple in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <cfconventions.org>`_.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. Must be compliant with the `CF Conventions <cfconventions.org>`_.
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate.
		units_z : `str`, optional
			Units for the :math:`z`-coordinate. Must be compliant with the `CF Conventions <cfconventions.org>`_.
		dims_z : `str`, optional
			Label for the :math:`z`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that :math:`z_F = z_T`, with :math:`z_T` 
			the value of :math:`z` at the top of the domain. In other words, a fully terrain-following coordinate
			system is supposed.
		topo_type : `str`, optional
			Topography type. Default is 'flat_terrain'. See :mod:`~tasmania.grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. See 
			:mod:`~tasmania.grids.topography` for further details.

		Keyword arguments
		-----------------
		kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~tasmania.grids.topography.Topography2d`.
		"""
		self.xy_grid = GridXY(domain_x, nx, domain_y, ny, units_x, dims_x, units_y, dims_y)

		z_hl = np.linspace(domain_z[0], domain_z[1], nz+1, dtype = datatype)
		self.z_on_interface_levels = Axis(z_hl, dims_z + '_half_levels', attrs = {'units': units_z})
		self.z = Axis(0.5 * (z_hl[:-1] + z_hl[1:]), dims_z, attrs = {'units': units_z})
		self.nz = int(nz)
		self.dz = math.fabs(domain_z[1] - domain_z[0]) / float(nz)

		if z_interface is None:
			z_interface = domain_z[0]
		if lt(domain_z[0], domain_z[1]):
			if not (le(domain_z[0], z_interface) and le(z_interface, domain_z[1])):
				raise ValueError('z_interface should be in the range(domain_z[0], domain_z[1]).')
		else:
			if not (le(domain_z[1], z_interface) and le(z_interface, domain_z[0])):
				raise ValueError('z_interface should be in the range(domain_z[1], domain_z[0]).')
		self.z_interface = z_interface 

		self.topography = Topography2d(self.xy_grid, topo_type, topo_time, **kwargs)

	@property
	def x(self):
		"""
		Get the :math:`x`-coordinates of the mass points.

		Return
		------
		obj :
			:class:`~tasmania.grids.axis.Axis` storing the :math:`x`-coordinates of the mass points.
		"""
		return self.xy_grid.x

	@property
	def x_at_u_locations(self):
		"""
		Get the :math:`x`-coordinates at the :math:`u`-locations.

		Return
		------
		obj :
			:class:`~tasmania.grids.axis.Axis` storing the :math:`x`-coordinates at the :math:`u`-locations.
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
		obj :
			:class:`~tasmania.grids.axis.Axis` storing the :math:`y`-coordinates of the mass points.
		"""
		return self.xy_grid.y

	@property
	def y_at_v_locations(self):
		"""
		Get the :math:`y`-coordinates at the :math:`v`-locations.

		Return
		------
		obj :
			:class:`~tasmania.grids.axis.Axis` storing the :math:`y`-coordinates at the :math:`v`-locations.
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
			Two-dimensional :class:`numpy.ndarray` representing the topography height.
		"""
		return self.topography.topo.values

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography. 

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self.topography.update(time)
