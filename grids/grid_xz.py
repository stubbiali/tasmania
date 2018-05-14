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
from tasmania.grids.topography import Topography1d
from tasmania.namelist import datatype
from tasmania.utils.utils import smaller_than as lt
from tasmania.utils.utils import smaller_or_equal_than as le

class GridXZ():
	"""
	Rectangular and regular two-dimensional grid embedded in a reference system whose coordinates are

		* the horizontal coordinate :math:`x`;
		* the vertical (terrain-following) coordinate :math:`z`. 	
	
	The vertical coordinate :math:`z` may be formulated to define a hybrid terrain-following coordinate system 
	with terrain-following coordinate lines between the surface terrain-height and :math:`z = z_F`, where 
	:math:`z`-coordinate lines change back to flat horizontal lines. However, no assumption is made on the actual 
	nature of :math:`z` which may be either pressure-based or height-based.

	Attributes
	----------
	x : obj
		:class:`~grids.axis.Axis` representing the :math:`x`-axis.
	nx : int
		Number of grid points along :math:`x`.
	dx : float
		The :math:`x`-spacing.
	z : obj
		:class:`~grids.axis.Axis` representing the :math:`z`-main levels.
	z_on_interface_levels : obj
		:class:`~grids.axis.Axis` representing the :math:`z`-half levels.
	nz : int
		Number of vertical main levels.
	dz : float
		The :math:`z`-spacing.
	z_interface : float
		The interface coordinate :math:`z_F`.

	Note
	----
	For the sake of compliancy with the `COSMO model <http://www.cosmo-model.org>`_, the vertical grid points are 
	ordered from the top of the domain to the surface.
	"""
	def __init__(self, domain_x, nx, domain_z, nz, 
				 units_x = 'm', dims_x = 'x', units_z = 'm', dims_z = 'z', z_interface = None,
				 topo_type = 'terrain_flat', topo_time = timedelta(), **kwargs):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{left}, ~ x_{right})`.
		nx : int
			Number of grid points in the :math:`x`-direction.
		domain_z : tuple
			Tuple in the form :math:`(z_{top}, ~ z_{surface})`.
		nz : int
			Number of vertical main levels.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`grids.axis.Axis.__init__`).
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_z : `str`, optional
			Units for the :math:`z`-coordinate. Must be compliant with the `CF Conventions <http://cfconventions.org>`_ 
			(see also :meth:`grids.axis.Axis.__init__`).
		dims_z : `str`, optional
			Label for the :math:`z`-coordinate.
		z_interface : `float`, optional
			Interface value :math:`z_F`. If not specified, it is assumed that :math:`z_F = z_T`, with :math:`z_T` the 
			value of :math:`z` at the top of the domain. In other words, a fully terrain-following coordinate system is 
			supposed.
		topo_type : `str`, optional
			Topography type. See :mod:`grids.topography` for further details.
		topo_time : `obj`, optional
			:class:`datetime.timedelta` representing the simulation time after which the topography should stop 
			increasing. Default is 0, corresponding to a time-invariant terrain surface-height. See 
			:mod:`grids.topography` for further details.

		Keyword arguments
		-----------------
		kwargs : 
			Keyword arguments to be forwarded to the constructor of :class:`~grids.topography.Topography1d`.
		"""
		self.x = Axis(np.linspace(domain_x[0], domain_x[1], nx, dtype = datatype),
					  dims_x, attrs = {'units': units_x}) 
		self.nx = int(nx)
		self.dx = (domain_x[1] - domain_x[0]) / (nx - 1.)

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

		self._topography = Topography1d(self.x, topo_type = topo_type, topo_time = topo_time, **kwargs)

	@property
	def topography_height(self):
		"""
		Get the topography (i.e., terrain-surface) height.

		Return
		------
		array_like : 
			One-dimensional :class:`numpy.ndarray` representing the topography height.
		"""
		return self._topography.topo.values

	def update_topography(self, time):
		"""
		Update the (time-dependent) topography.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self._topography.update(time)

