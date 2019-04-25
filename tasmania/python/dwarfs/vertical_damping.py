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
	Rayleigh(VerticalDamping)
"""
import abc
import math
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.utils.utils import greater_or_equal_than as ge

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class VerticalDamping:
	"""
	Abstract base class whose derived classes implement different
	vertical damping, i.e., wave absorbing, techniques.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, shape, grid, damp_depth, damp_coeff_max,
		time_units, backend, dtype
	):
		"""
		Parameters
		----------
		shape : tuple
			Shape of the 3-D arrays on which applying the absorber.
		grid : tasmania.Grid
			The underlying grid.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_coeff_max : float
			Maximum value for the damping coefficient.
		time_units : str
			Time units to be used throughout the class.
		backend : obj
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		# safety-guard checks
		assert damp_depth <= grid.nz, \
			"The depth of the damping region ({}) should be smaller or equal than " \
			"the number of main vertical levels ({}).".format(damp_depth, grid.nz)

		# store input arguments
		self._shape   = shape
		self._grid    = grid
		self._depth   = damp_depth
		self._cmax    = damp_coeff_max
		self._tunits  = time_units
		self._backend = backend

		# compute lower-bound of damping region
		self._lb = grid.z.values[damp_depth-1]

		# initialize the underlying GT4Py stencil
		self._stencil_initialize(dtype)

	@abc.abstractmethod
	def __call__(
		self, dt, field_now, field_new, field_ref, field_out
	):
		"""
		Apply vertical damping to a generic field.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		dt : timedelta
			The time step.
		field_now : numpy.ndarray
			The field at the current time level.
		field_new : numpy.ndarray
			The field at the next time level, on which the absorber will be applied.
		field_ref : numpy.ndarray
			A reference value for the field.
		field_out : array_like
			Buffer into which writing the output, vertically damped field.
		"""

	@abc.abstractmethod
	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying vertical damping.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""

	@staticmethod
	def factory(
		damp_type, shape, grid, damp_depth, damp_coeff_max,
		time_units='s', backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Static method which returns an instance of the derived class
		implementing the damping method specified by :data:`damp_type`.

		Parameters
		----------
		damp_type : str
			String specifying the damper to implement. Either:

				* 'rayleigh', for a Rayleigh damper.

		shape : tuple
			Shape of the 3-D arrays on which applying the absorber.
		grid : tasmania.Grid
			The underlying grid.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_coeff_max : float
			Maximum value for the damping coefficient.
		time_units : `str`, optional
			Time units to be used throughout the class. Defaults to 's'.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.

		Return
		------
		obj :
			An instance of the appropriate derived class.
		"""
		args = [shape, grid, damp_depth, damp_coeff_max, time_units, backend, dtype]
		if damp_type == 'rayleigh':
			return Rayleigh(*args)
		else:
			raise ValueError('Unknown damping scheme. Available options: ''rayleigh''.')


class Rayleigh(VerticalDamping):
	"""
	This class inherits	:class:`~tasmania.VerticalDamping`
	to implement a Rayleigh absorber.
	"""
	def __init__(
		self, shape, grid, damp_depth=15, damp_coeff_max=0.0002,
		time_units='s', backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Parameters
		----------
		shape : tuple
			Shape of the 3-D arrays on which applying the absorber.
		grid : tasmania.Grid
			The underlying grid.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_coeff_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		time_units : `str`, optional
			Time units to be used throughout the class. Defaults to 's'.
		backend : `obj`, optional
			TODO
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""
		super().__init__(
			shape, grid, damp_depth, damp_coeff_max, time_units, backend, dtype
		)

	def __call__(self, dt, field_now, field_new, field_ref, field_out):
		# update the attributes which will serve as stencil's inputs
		self._dt.value = \
			DataArray(dt.total_seconds(), attrs={'units': 's'}).to_units(self._tunits).values.item()
		self._field_now[...] = field_now[...]
		self._field_new[...] = field_new[...]
		self._field_ref[...] = field_ref[...]

		# run the stencil's compute function
		self._stencil.compute()

		# write into the output buffer
		field_out[:, :, :self._depth] = self._field_out[:, :, :self._depth]
		field_out[:, :, self._depth:] = field_new[:, :, self._depth:]

	def _stencil_initialize(self, dtype):
		# shortcuts
		grid = self._grid
		nz, za, zt = grid.nz, self._lb, grid.z_on_interface_levels.values[0]
		ni, nj, nk = self._shape

		if nk == nz:
			# compute the damping matrix in the case of a z-unstaggered field
			z = grid.z.values
			r = ge(z, za) * self._cmax * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))
		else:
			# compute the damping matrix in the case of a z-staggered field
			z = grid.z_on_interface_levels.values
			r = ge(z, za) * self._cmax * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))

		# allocate the attributes which will serve as stencil's inputs
		self._dt = gt.Global()
		self._field_now = np.zeros((ni, nj, nk), dtype=dtype)
		self._field_new = np.zeros((ni, nj, nk), dtype=dtype)
		self._field_ref = np.zeros((ni, nj, nk), dtype=dtype)

		# allocate the numpy array which will serve as stencil's output
		self._field_out = np.zeros((ni, nj, nk), dtype=dtype)

		# instantiate the stencil
		# TODO: exclude boundary layers from stencil computational domain
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'field_now': self._field_now, 'field_new': self._field_new,
				'field_ref': self._field_ref, 'rmat': self._rmat
			},
			global_inputs={'dt': self._dt},
			outputs={'field_out': self._field_out},
			domain=gt.domain.Rectangle((0, 0, 0), (ni-1, nj-1, self._depth-1)),
			mode=self._backend
		)

	@staticmethod
	def _stencil_defs(dt, field_now, field_new, field_ref, rmat):
		"""
		The GT4Py stencil applying Rayleigh vertical damping.

		Parameters
		----------
		dt : gridtools.Global
			The time step.
		field_now : gridtools.Equation
			The field at the current time level.
		field_new : gridtools.Equation
			The field at the next time level, on which the absorber will be applied.
		field_ref : gridtools.Equation
			The reference field.
		rmat : gridtools.Equation
			The damping coefficient.

		Return
		------
		gridtools.Equation :
			The damped field.
		"""
		# indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# output field
		field_out = gt.Equation()

		# computations
		field_out[i, j, k] = field_new[i, j, k] - \
			dt * rmat[i, j, k] * (field_now[i, j, k] - field_ref[i, j, k])

		return field_out
