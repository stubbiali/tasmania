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

from tasmania.grids.topography import Topography1d
from tasmania.namelist import datatype
from tasmania.utils.utils import smaller_than as lt, \
								 smaller_or_equal_than as le


class GridXZ:
	"""
	This class represents a rectangular and regular two-dimensional grid
	embedded in a reference system whose coordinates are

		* the horizontal coordinate :math:`x`;
		* the vertical (terrain-following) coordinate :math:`z`. 	
	
	The vertical coordinate :math:`z` may be formulated to define a
	hybrid terrain-following coordinate system with terrain-following
	coordinate lines between the surface terrain-height and :math:`z = z_F`,
	where :math:`z`-coordinate lines change back to flat horizontal lines.
	However, no assumption is made on the actual nature of :math:`z`,
	which may be either pressure-based or height-based.

	Attributes
	----------
	x : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the mass points.
	x_at_u_locations : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the :math:`x`-staggered points.
	nx : int
		Number of mass points in the :math:`x`-direction.
	dx : float
		The :math:`x`-spacing.
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
		:class:`~tasmania.grids.topography.Topography1d` representing
		the underlying topography.

	Note
	----
	For the sake of compliancy with the `COSMO model <http://www.cosmo-model.org>`_,
	the vertical grid points are ordered from the top of the domain to the surface.
	"""
	def __init__(self, domain_x, nx, domain_z, nz, 
				 dims_x='x', units_x='m', dims_z='z', units_z='m',
				 z_interface=None, dtype=datatype,
				 topo_type='flat_terrain', topo_time=timedelta(), topo_kwargs=None):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			The interval which the domain includes along the :math:`x`-axis.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_z : tuple
			The interval which the domain includes along the :math:`z`-axis.
			This should be specified in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate. Defaults to 'x'.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'm'.
		dims_z : `str`, optional
			Label for the :math:`z`-coordinate. Defaults to 'z'.
		units_z : `str`, optional
			Units for the :math:`z`-coordinate.
			Should be compliant with the `CF Conventions <http://cfconventions.org>`_.
			Defaults to 'm'.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that
			:math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the top
			of the domain. In other words, the coordinate system is supposed
			fully terrain-following.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.
		topo_type : `str`, optional
			Topography type. Defaults to 'flat_terrain'.
			See :class:`~tasmania.grids.topography.Topography1d` for further details.
		topo_time : `timedelta`, optional
			:class:`datetime.timedelta` representing the simulation time after
			which the topography should stop increasing. Default is 0, corresponding
			to a time-invariant terrain surface-height. See
			:mod:`~tasmania.grids.topography.Topography1d` for further details.
		topo_kwargs : `dict`, optional
			Keyword arguments to be forwarded to the constructor of
			:class:`~tasmania.grids.topography.Topography1d`.

		Raises
		------
		ValueError :
			If :obj:`z_interface` is outside the domain.
		"""
		# x-coordinates of the mass points
		xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
			 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
		self.x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x',
								 attrs={'units': units_x})

		# Number of mass points along the x-axis
		self.nx = int(nx)

		# x-spacing
		self.dx = (domain_x[1]-domain_x[0]) / (nx-1.)

		# x-coordinates of the x-staggered points
		xv_u = np.linspace(domain_x[0] - 0.5*self.dx, domain_x[1] + 0.5*self.dx,
						   nx+1, dtype=dtype)
		self.x_at_u_locations = sympl.DataArray(xv_u, coords=[xv_u], dims=dims_x,
												name='x_at_u_locations',
												attrs={'units': units_x})

		# z-coordinates of the half-levels
		zv_hl = np.linspace(domain_z[0], domain_z[1], nz+1, dtype=dtype)
		self.z_on_interface_levels = sympl.DataArray(zv_hl, coords=[zv_hl],
													 dims=dims_z,
													 name='z_on_interface_levels',
													 attrs={'units': units_z})

		# z-coordinates of the main-levels
		zv = 0.5 * (zv_hl[:-1] + zv_hl[1:])
		self.z = sympl.DataArray(zv, coords=[zv], dims=dims_z, name='z',
								 attrs={'units': units_z})

		# Number of vertical main levels
		self.nz = int(nz)

		# z-spacing
		self.dz = math.fabs(domain_z[1] - domain_z[0]) / float(nz)

		# z-interface
		if z_interface is None:
			self.z_interface = domain_z[0]
		else:
			self.z_interface = z_interface
		if lt(domain_z[0], domain_z[1]):
			if not (le(domain_z[0], self.z_interface) and
					le(self.z_interface, domain_z[1])):
				raise ValueError('z_interface should be in the range '
								 '({}, {}).'.format(domain_z[0], domain_z[1]))
		else:
			if not (le(domain_z[1], self.z_interface) and
					le(self.z_interface, domain_z[0])):
				raise ValueError('z_interface should be in the range '
								 '({}, {}).'.format(domain_z[1], domain_z[0]))

		# Underlying topography
		if topo_kwargs is None or not isinstance(topo_kwargs, dict):
			self.topography = Topography1d(self.x, topo_type=topo_type)
		else:
			self.topography = Topography1d(self.x, topo_type=topo_type,
									   	   topo_time=topo_time, **topo_kwargs)

	@property
	def topography_height(self):
		"""
		Get the topography (i.e., terrain-surface) height.

		Return
		------
		array_like : 
			1-D :class:`numpy.ndarray` representing the topography height.
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
