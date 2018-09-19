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
	VerticalDamping
	_Rayleigh(VerticalDamping)
"""
import abc
import math
import numpy as np

import gridtools as gt
from tasmania.utils.utils import greater_or_equal_than as ge
try:
	from tasmania.namelist import datatype
except ImportError:
	from numpy import float32 as datatype


class VerticalDamping:
	"""
	Abstract base class whose derived classes implement different
	vertical damping, i.e., wave absorbing, techniques.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, dims, grid, damp_depth, damp_max, backend, dtype):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Shape of the (three-dimensional) arrays on which to apply
			vertical damping.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_max : float
			Maximum value for the damping coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the
			underlying GT4Py stencil
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""
		# Store input arguments
		self._dims       = dims
		self._grid       = grid
		self._damp_depth = damp_depth
		self._damp_max   = damp_max
		self._backend    = backend

		# Compute lower-bound of damping region
		self._damp_lb = self._grid.z.values[damp_depth-1]

		# Initialize the underlying GT4Py stencil
		self._stencil_initialize(dtype)

	@abc.abstractmethod
	def __call__(self, dt, phi_now, phi_new, phi_ref, phi_out):
		"""
		Apply vertical damping to a generic field :math:`\phi`. 
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		dt : timedelta
			:class:`datetime.timedelta` representing the time step.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi`
			at the current time level.
		phi_new : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi`
			at the next time level, on which the absorber will be applied.
		phi_ref : array_like
			:class:`numpy.ndarray` representing a reference value for
			:math:`\phi`.
		phi_out : array_like
			:class:`numpy.ndarray` into which writing the output,
			vertically damped field.
		"""

	@abc.abstractmethod
	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal damping.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""

	@staticmethod
	def factory(damp_type, dims, grid, damp_depth, damp_max,
				backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Static method which returns an instance of the derived class
		implementing the damping method specified by :data:`damp_type`.

		Parameters
		----------
		damp_type : str
			String specifying the damper to implement. Either:

				* 'rayleigh', for a Rayleigh damper.

		dims : tuple
			Shape of the (three-dimensional) arrays on which to apply
			vertical damping.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_max : float
			Maximum value for the damping coefficient.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the
			underlying GT4Py stencil. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.

		Return
		------
		obj :
			An instance of the derived class implementing the damping method
			specified by :data:`damp_type`.
		"""
		if damp_type == 'rayleigh':
			return _Rayleigh(dims, grid, damp_depth, damp_max, backend, dtype)
		else:
			raise ValueError('Unknown damping scheme. Available options: ''rayleigh''.')


class _Rayleigh(VerticalDamping):
	"""
	This class inherits
	:class:`~tasmania.dynamics.vertical_damping.VerticalDamping`
	to implement a Rayleigh absorber.
	"""
	def __init__(self, dims, grid, damp_depth=15, damp_max=0.0002,
				 backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Shape of the (three-dimensional) arrays on which to apply
			vertical damping.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the
			underlying GT4Py stencil. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(dims, grid, damp_depth, damp_max, backend, dtype)

	def __call__(self, dt, phi_now, phi_new, phi_ref, phi_out):
		"""
		Apply vertical damping to a generic field :math:`\phi`. 
		"""
		# Update the attributes which will serve as stencil's inputs
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		self._phi_now[...] = phi_now[...]
		self._phi_new[...] = phi_new[...]
		self._phi_ref[...] = phi_ref[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Write to the passed array
		phi_out[:, :, :self._damp_depth] = self._phi_out[:, :, :self._damp_depth]
		phi_out[:, :, self._damp_depth:] = phi_new[:, :, self._damp_depth:]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying Rayleigh vertical damping.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		ni, nj, nk = self._dims
		za, zt = self._damp_lb, self._grid.z_on_interface_levels.values[0]

		if nk == nz:
			# Compute the damping matrix which should be used in case
			# of a z-unstaggered field
			z = self._grid.z.values
			r = ge(z, za) * self._damp_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))
		else:
			# Compute the damping matrix which should be used in case
			# of a z-staggered field
			z = self._grid.z_on_interface_levels.values
			r = ge(z, za) * self._damp_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))

		# Allocate the attributes which will serve as stencil's inputs
		self._dt = gt.Global()
		self._phi_now = np.zeros((ni, nj, nk), dtype=dtype)
		self._phi_new = np.zeros((ni, nj, nk), dtype=dtype)
		self._phi_ref = np.zeros((ni, nj, nk), dtype=dtype)

		# Allocate the Numpy array which will serve as stencil's output
		self._phi_out = np.zeros((ni, nj, nk), dtype=dtype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'phi_now': self._phi_now, 'phi_new': self._phi_new,
								'phi_ref': self._phi_ref, 'rmat': self._rmat},
			global_inputs	 = {'dt': self._dt},
			outputs			 = {'phi_out': self._phi_out},
			domain			 = gt.domain.Rectangle((0, 0, 0),
												   (ni-1, nj-1, self._damp_depth-1)),
			mode			 = self._backend)

	@staticmethod
	def _stencil_defs(dt, phi_now, phi_new, phi_ref, rmat):
		"""
		The GT4Py stencil applying Rayleigh vertical damping.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		phi_now : ibj
			:class:`gridtools.Equation` representing the field :math:`\phi`
			at the current time level.
		phi_new : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`
			at the next time level, on which the absorber will be applied.
		phi_ref : obj
			:class:`gridtools.Equation` representing a reference value for
			:math:`\phi`.
		rmat : obj
			:class:`gridtools.Equation` representing the damping coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the damped field :math:`\phi`.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		phi_out = gt.Equation()

		# Computations
		phi_out[i, j, k] = phi_new[i, j, k] - \
						   dt * rmat[i, j, k] * (phi_now[i, j, k] - phi_ref[i, j, k])

		return phi_out
