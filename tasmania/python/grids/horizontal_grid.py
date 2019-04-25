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
This module contains:
	HorizontalGrid
	PhysicalHorizontalGrid
	NumericalHorizontalGrid
"""
import numpy as np
import sympl

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class HorizontalGrid:
	"""
	This class represents a rectangular and regular two-dimensional grid
	embedded in a reference	system whose coordinates are, in the order,
	:math:`x` and :math:`y`. No assumption is made on the nature of the
	coordinates. For instance, :math:`x` may be the longitude - in which
	case :math:`x \equiv \lambda` - and :math:`y` may be the latitude -
	in which case :math:`y \equiv \phi`.
	"""
	def __init__(self, x, y, x_at_u_locations=None, y_at_v_locations=None):
		"""
		Parameters
		----------
		x : sympl.DataArray
			1-D :class:`sympl.DataArray` collecting the mass grid points
			along the first horizontal dimension.
		y : sympl.DataArray
			1-D :class:`sympl.DataArray` collecting the mass grid points
			along the second horizontal dimension.
		x_at_u_locations : `sympl.DataArray`, optional
			1-D :class:`sympl.DataArray` collecting the staggered grid points
			along the first horizontal dimension. If not given, this is
			retrieved from `x`.
		y_at_v_locations : `sympl.DataArray`, optional
			1-D :class:`sympl.DataArray` collecting the staggered grid points
			along the second horizontal dimension. If not given, this is
			retrieved from `y`.
		"""
		# global properties
		dtype = x.values.dtype

		# x-coordinates of the mass points
		self._x = x

		# number of mass points along the x-axis
		nx = x.values.shape[0]
		self._nx = nx

		# x-spacing
		dx_v = 1.0 if nx == 1 else (x.values[-1]-x.values[0]) / (nx-1)
		dx_v = dx_v if dx_v != 0.0 else 1.0
		self._dx = sympl.DataArray(dx_v, name='dx', attrs={'units': x.attrs['units']})

		# x-coordinates of the x-staggered points
		if x_at_u_locations is not None:
			self._xu = x_at_u_locations
		else:
			xu_v = np.linspace(
				x.values[0] - 0.5*dx_v, x.values[-1] + 0.5*dx_v, nx+1, dtype=dtype
			)
			self._xu = sympl.DataArray(
				xu_v, coords=[xu_v], dims=(x.dims[0] + '_at_u_locations'),
				name=x.dims[0] + '_at_u_locations', attrs={'units': x.attrs['units']}
			)

		# y-coordinates of the mass points
		self._y = y

		# number of mass points along the y-axis
		ny = self._y.values.shape[0]
		self._ny = ny

		# y-spacing
		dy_v = 1.0 if ny == 1 else (y.values[-1]-y.values[0]) / (ny-1)
		dy_v = dy_v if dy_v != 0.0 else 1.0
		self._dy = sympl.DataArray(dy_v, name='dy', attrs={'units': y.attrs['units']})

		# y-coordinates of the y-staggered points
		if y_at_v_locations is not None:
			self._yv = y_at_v_locations
		else:
			yv_v = np.linspace(
				y.values[0] - 0.5*dy_v, y.values[-1] + 0.5*dy_v, ny+1, dtype=dtype
			)
			self._yv = sympl.DataArray(
				yv_v, coords=[yv_v], dims=(y.dims[0] + '_at_v_locations'),
				name=y.dims[0] + '_at_v_locations', attrs={'units': y.attrs['units']}
			)

	@property
	def x(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-D :class:`sympl.DataArray` collecting the mass grid points
			along the first horizontal dimension.
		"""
		return self._x

	@property
	def x_at_u_locations(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-D :class:`sympl.DataArray` collecting the staggered grid points
			along the first horizontal dimension.
		"""
		return self._xu

	@property
	def nx(self):
		"""
		Returns
		-------
		int :
			Number of mass grid points featured by the grid along
			the first horizontal dimension.
		"""
		return self._nx

	@property
	def dx(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-item :class:`sympl.DataArray` representing the grid spacing
			along the first horizontal dimension.
		"""
		return self._dx

	@property
	def y(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-D :class:`sympl.DataArray` collecting the mass grid points
			along the second horizontal dimension.
		"""
		return self._y

	@property
	def y_at_v_locations(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-D :class:`sympl.DataArray` collecting the staggered grid points
			along the second horizontal dimension.
		"""
		return self._yv

	@property
	def ny(self):
		"""
		Returns
		-------
		int :
			Number of mass grid points featured by the grid along
			the second horizontal dimension.
		"""
		return self._ny

	@property
	def dy(self):
		"""
		Returns
		-------
		sympl.DataArray :
			1-item :class:`sympl.DataArray` representing the grid spacing
			along the second horizontal dimension.
		"""
		return self._dy


class PhysicalHorizontalGrid(HorizontalGrid):
	"""
	This class represents a rectangular and regular grid embedded
	in a two-dimensional *physical* domain.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, dtype=datatype):
		""" 
		Parameters
		----------
		domain_x : sympl.DataArray
			2-items :class:`sympl.DataArray` storing the end-points, dimension
			and units of the interval which the physical domain includes along
			the first horizontal dimension.
		nx : int
			Number of mass points featured by the grid along the first
			horizontal dimension.
		domain_y : sympl.DataArray
			2-items :class:`sympl.DataArray` storing the end-points, dimension
			and units of the interval which the physical domain includes along
			the second horizontal dimension.
		ny : int
			Number of mass points featured by the grid along the second
			horizontal dimension.
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated within this class.

		Note
		----
		Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
		"""
		# extract x-axis properties
		values_x = domain_x.values
		dims_x   = domain_x.dims
		units_x  = domain_x.attrs['units']

		# x-coordinates of the mass points
		x_v = np.array([0.5 * (values_x[0]+values_x[1])], dtype=dtype) if nx == 1 \
			else np.linspace(values_x[0], values_x[1], nx, dtype=dtype)
		x = sympl.DataArray(
			x_v, coords=[x_v], dims=dims_x, name=dims_x[0], attrs={'units': units_x}
		)

		# extract y-axis properties
		values_y = domain_y.values
		dims_y   = domain_y.dims
		units_y  = domain_y.attrs['units']

		# y-coordinates of the mass points
		y_v = np.array([0.5 * (values_y[0]+values_y[1])], dtype=dtype) if ny == 1 \
			else np.linspace(values_y[0], values_y[1], ny, dtype=dtype)
		y = sympl.DataArray(
			y_v, coords=[y_v], dims=dims_y, name=dims_y[0], attrs={'units': units_y}
		)

		# call parent's constructor
		super().__init__(x, y)


class NumericalHorizontalGrid(HorizontalGrid):
	"""
	This class represents a rectangular and regular grid embedded
	in a two-dimensional *numerical* domain.
	"""
	def __init__(self, phys_grid, boundary):
		"""
		Parameters
		----------
		phys_grid : tasmania.HorizontalGrid
			The associated *physical* grid.
		boundary : tasmania.HorizontalBoundary
			The :class:`tasmania.HorizontalBoundary` handling the horizontal
			boundary conditions.
		"""
		# x-coordinates of the mass points
		dims = 'c_' + phys_grid.x.dims[0]
		x = boundary.get_numerical_xaxis(phys_grid.x, dims=dims)

		# x-coordinates of the x-staggered points
		dims = 'c_' + phys_grid.x_at_u_locations.dims[0]
		xu = boundary.get_numerical_xaxis(phys_grid.x_at_u_locations, dims=dims)

		# y-coordinates of the mass points
		dims = 'c_' + phys_grid.y.dims[0]
		y = boundary.get_numerical_yaxis(phys_grid.y, dims=dims)

		# y-coordinates of the y-staggered points
		dims = 'c_' + phys_grid.y_at_v_locations.dims[0]
		yv = boundary.get_numerical_yaxis(phys_grid.y_at_v_locations, dims=dims)

		# call parent's constructor
		super().__init__(x, y, xu, yv)

		# coherency checks
		assert self.nx == boundary.ni
		assert self.ny == boundary.nj
