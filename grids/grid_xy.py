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
import math
import matplotlib.pyplot as plt
import numpy as np

from tasmania.grids.axis import Axis
from tasmania.namelist import datatype

class GridXY:
	"""
	Rectangular and regular two-dimensional grid embedded in a reference system whose coordinates are, 
	in the order, :math:`x` and :math:`y`. No assumption is made on the nature of the coordinates. For 
	instance, :math:`x` may be the longitude, in which case :math:`x \equiv \lambda`, and :math:`y` may 
	be the latitude, in which case :math:`y \equiv \phi`.

	Attributes
	----------
	x : obj
		:class:`~tasmania.grids.axis.Axis` storing the :math:`x`-coordinates of the mass points.
	x_at_u_locations : obj
		:class:`~tasmania.grids.axis.Axis` storing the :math:`x`-coordinates at the :math:`u`-locations.
	nx : int
		Number of grid points along :math:`x`.
	dx : float
		The :math:`x`-spacing.
	y : obj
		:class:`~tasmania.grids.axis.Axis` storing the :math:`y`-coordinates of the mass points.
	y_at_v_locations : obj
		:class:`~tasmania.grids.axis.Axis` storing the :math:`y`-coordinates at the :math:`v`-locations.
	ny : int
		Number of grid points along :math:`y`.
	dy : float
		The :math:`y`-spacing.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, 
				 units_x = 'degrees_east', dims_x = 'longitude', units_y = 'degrees_north', dims_y = 'latitude'):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{start}, ~ x_{stop})`.
		nx : int
			Number of grid points along :math:`x`.
		domain_y : tuple
			Tuple in the form :math:`(y_{start}, ~ y_{stop})`.
		ny : int
			Number of grid points along :math:`y`.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. 
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate.

		Note
		----
		Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
		"""
		xv = np.array([.5 * (domain_x[0] + domain_x[1])], dtype = datatype) if nx == 1 \
			 else np.linspace(domain_x[0], domain_x[1], nx, dtype = datatype)
		self.x = Axis(xv, dims_x, attrs = {'units': units_x}) 
		self.nx = int(nx)
		self.dx = 1. if nx == 1 else (domain_x[1] - domain_x[0]) / (nx - 1.)
		self.x_at_u_locations = Axis(
			np.linspace(domain_x[0] - 0.5 * self.dx, domain_x[1] + 0.5 * self.dx, nx + 1, dtype = datatype),
			dims_x, attrs = {'units': units_x})

		yv = np.array([.5 * (domain_y[0] + domain_y[1])], dtype = datatype) if ny == 1 \
			 else np.linspace(domain_y[0], domain_y[1], ny, dtype = datatype)
		self.y = Axis(yv, dims_y, attrs = {'units': units_y})
		self.ny = int(ny)
		self.dy = 1. if ny == 1 else (domain_y[1] - domain_y[0]) / (ny - 1.)
		self.y_at_v_locations = Axis(
			np.linspace(domain_y[0] - 0.5 * self.dy, domain_y[1] + 0.5 * self.dy, ny + 1, dtype = datatype),
			dims_y, attrs = {'units': units_y})
