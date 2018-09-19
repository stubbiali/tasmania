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
	GridXY
"""
import numpy as np
import sympl

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class GridXY:
	"""
	This class represents a rectangular and regular two-dimensional grid
	embedded in a reference system whose coordinates are, in the order,
	:math:`x` and :math:`y`. No assumption is made on the nature of the
	coordinates. For instance, :math:`x` may be the longitude, in which
	case :math:`x \equiv \lambda`, and :math:`y` may be the latitude,
	in which case :math:`y \equiv \phi`.

	Attributes
	----------
	x : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the mass points.
	x_at_u_locations : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the :math:`staggered` points.
	nx : int
		Number of mass points in the :math:`x`-direction.
	dx : dataarray_like
		The :math:`x`-spacing, in the same units of the :math:`x`-axis.
	y : dataarray_like
		:class:`sympl.DataArray` storing the :math:`y`-coordinates of
		the mass points.
	y_at_v_locations : dataarray_like
		:class:`sympl.DataArray` storing the :math:`y`-coordinates of
		the :math:`y`-staggered points.
	ny : int
		Number of mass points in the :math:`y`-direction.
	dy : dataarray_like
		The :math:`y`-spacing, in the same units of the :math:`y`-axis.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, dtype=datatype):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : dataarray_like
			2-items :class:`sympl.DataArray` storing the end-points of the interval
			which the domain includes along the :math:`x`-axis, as well as the axis
			dimension and units.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_y : dataarray_like
			2-items :class:`sympl.DataArray` storing the end-points of the interval
			which the domain includes along the :math:`y`-axis, as well as the axis
			dimension and units.
		ny : int
			Number of mass points in the :math:`y`-direction.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.

		Note
		----
		Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
		"""
		# Extract x-axis properties
		values_x = domain_x.values
		dims_x   = domain_x.dims
		units_x  = domain_x.attrs['units']

		# x-coordinates of the mass points
		xv = np.array([0.5 * (values_x[0]+values_x[1])], dtype=dtype) if nx == 1 \
			 else np.linspace(values_x[0], values_x[1], nx, dtype=dtype)
		self.x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name=dims_x[0],
								 attrs={'units': units_x})

		# Number of mass points along the x-axis
		self.nx = int(nx)

		# x-spacing
		dx_v = 1. if nx == 1 else (values_x[1]-values_x[0]) / (nx-1.)
		self.dx = sympl.DataArray(dx_v, name='dx', attrs={'units': units_x})

		# x-coordinates of the x-staggered points
		xv_u = np.linspace(values_x[0] - 0.5*dx_v, values_x[1] + 0.5*dx_v,
						   nx+1, dtype=dtype)
		self.x_at_u_locations = sympl.DataArray(xv_u, coords=[xv_u],
												dims=(dims_x[0] + '_at_u_locations'),
												name=dims_x[0] + '_at_u_locations',
												attrs={'units': units_x})

		# Extract y-axis properties
		values_y = domain_y.values
		dims_y   = domain_y.dims
		units_y  = domain_y.attrs['units']

		# y-coordinates of the mass points
		yv = np.array([0.5 * (values_y[0]+values_y[1])], dtype=dtype) if ny == 1 \
			 else np.linspace(values_y[0], values_y[1], ny, dtype=dtype)
		self.y = sympl.DataArray(yv, coords=[yv], dims=dims_y, name=dims_y[0],
								 attrs={'units': units_y})

		# Number of mass points along the y-axis
		self.ny = int(ny)

		# y-spacing
		dy_v = 1. if ny == 1 else (values_y[1]-values_y[0]) / (ny-1.)
		self.dy = sympl.DataArray(dy_v, name='dy', attrs={'units': units_y})

		# y-coordinates of the y-staggered points
		yv_v = np.linspace(values_y[0] - 0.5*dy_v, values_y[1] + 0.5*dy_v,
						   ny+1, dtype=dtype)
		self.y_at_v_locations = sympl.DataArray(yv_v, coords=[yv_v],
												dims=(dims_y[0] + '_at_v_locations'),
												name=dims_y[0] + '_at_v_locations',
												attrs={'units': units_y})
