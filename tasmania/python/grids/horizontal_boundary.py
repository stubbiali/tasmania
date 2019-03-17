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
	HorizontalBoundary
"""
import abc


class HorizontalBoundary:
	"""
	Abstract base class whose children handle the
	horizontal boundary conditions.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, nx, ny, nb):
		"""
		Parameters
		----------
		nx : int
			Number of mass points featured by the *physical* grid
			along the first horizontal dimension.
		ny : int
			Number of mass points featured by the *physical* grid
			along the second horizontal dimension.
		nb : int
			Number of boundary layers.
		"""
		self._nx = nx
		self._ny = ny
		self._nb = nb
		self._ref_state = None

	@property
	def nx(self):
		"""
		Return
		------
		int :
			Number of mass points featured by the *physical* grid
			along the first horizontal dimension.
		"""
		return self._nx

	@property
	def ny(self):
		"""
		Return
		------
		int :
			Number of mass points featured by the *physical* grid
			along the second horizontal dimension.
		"""
		return self._ny

	@property
	def nb(self):
		"""
		Return
		------
		int :
			Number of boundary layers.
		"""
		return self._nb

	@property
	@abc.abstractmethod
	def ni(self):
		"""
		Return
		------
		int :
			Number of mass points featured by the *computational* grid
			along the first horizontal dimension.
		"""
		pass

	@property
	@abc.abstractmethod
	def nj(self):
		"""
		Return
		------
		int :
			Number of mass points featured by the *computational* grid
			along the second horizontal dimension.
		"""
		pass

	@property
	def reference_state(self):
		"""
		Return
		------
		dict :
			The reference model state dictionary, defined over the
			computational grid.
		"""
		return self._ref_state if self._ref_state is not None else {}

	@reference_state.setter
	def reference_state(self, ref_state):
		"""
		Parameters
		----------
		ref_state : dict
			The reference model state dictionary.
		"""
		for name in ref_state:
			if name != 'time':
				assert 'units' in ref_state[name].attrs, \
					"Field {} of reference state misses units attribute.".format(name)

		self._ref_state = ref_state

	@abc.abstractmethod
	def get_computational_xaxis(self, paxis, dims=None):
		"""
		Parameters
		----------
		paxis : dataarray_like
			1-D :class:`sympl.DataArray` representing the axis along the
			first horizontal dimension of the physical domain.
			Both unstaggered and staggered grid locations are supported.
		dims : `str`, optional
			The dimension of the returned axis. If not specified, the
			returned axis will have the same dimension of the input axis.

		Return
		------
		dataarray_like :
			1-D :class:`sympl.DataArray` representing the associated
			computational axis.
		"""
		pass

	@abc.abstractmethod
	def get_computational_yaxis(self, paxis, dims=None):
		"""
		Parameters
		----------
		paxis : dataarray_like
			1-D :class:`sympl.DataArray` representing the axis along the
			second horizontal dimension of the physical domain.
			Both unstaggered and staggered grid locations are supported.
		dims : `str`, optional
			The dimension of the returned axis. If not specified, the
			returned axis will have the same dimension of the input axis.

		Return
		------
		dataarray_like :
			1-D :class:`sympl.DataArray` representing the associated
			computational axis.
		"""
		pass

	@abc.abstractmethod
	def get_computational_field(self, field, field_name=None):
		"""
		Parameters
		----------
		field : numpy.ndarray
			A field defined over the *physical* grid.
		field_name : `str`, optional
			Field name.

		Return
		------
		numpy.ndarray :
			The same field defined over the *computational* grid.
		"""

	@abc.abstractmethod
	def get_computational_state(self, state):
		"""
		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`sympl.DataArray`\s representing the
			fields of those variables defined over the *physical* grid.

		Return
		------
		dict :
			The same input model state, but defined over the *computational* grid.
		"""
		pass

	@abc.abstractmethod
	def get_physical_xaxis(self, caxis, dims=None):
		"""
		Parameters
		----------
		caxis : dataarray_like
			1-D :class:`sympl.DataArray` representing the axis along the
			first horizontal dimension of the computational domain.
			Both unstaggered and staggered grid locations are supported.
		dims : `str`, optional
			The dimension of the returned axis. If not specified, the
			returned axis will have the same dimension of the input axis.

		Return
		------
		dataarray_like :
			1-D :class:`sympl.DataArray` representing the associated
			physical axis.
		"""
		pass

	@abc.abstractmethod
	def get_physical_yaxis(self, caxis, dims=None):
		"""
		Parameters
		----------
		caxis : dataarray_like
			1-D :class:`sympl.DataArray` representing the axis along the
			second horizontal dimension of the computational domain.
			Both unstaggered and staggered grid locations are supported.
		dims : `str`, optional
			The dimension of the returned axis. If not specified, the
			returned axis will have the same dimension of the input axis.

		Return
		------
		dataarray_like :
			1-D :class:`sympl.DataArray` representing the associated
			physical axis.
		"""
		pass

	@abc.abstractmethod
	def get_physical_field(self, field):
		"""
		Parameters
		----------
		field : array_like
			:class:`numpy.ndarray` representing a field defined over
			the *computational* grid.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the same field defined
			over the *physical* grid.
		"""

	@abc.abstractmethod
	def get_physical_state(self, state):
		"""
		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`sympl.DataArray`\s representing the
			fields of those variables defined over the *computational* grid.

		Return
		------
		dict :
			The same input model state, but defined over the *physical* grid.
		"""

	@abc.abstractmethod
	def enforce_field(self, field, field_name=None, field_units=None, time=None):
		"""
		Enforce the horizontal boundary conditions on the passed field,
		which is modified in-place.

		Parameters
		----------
		field : array_like
			:class:`numpy.ndarray` collecting the raw field values.
		field_name : `str`, optional
			The field name.
		field_units : `str`, optional
			The field units.
		time : `datetime`, optional
			Temporal instant at which the field is defined.
		"""

	def enforce_raw(self, state, field_properties=None):
		"""
		Enforce the horizontal boundary conditions on the passed state,
		which is modified in-place.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`numpy.ndarray`\s
			storing values for those variables.
		field_properties : `dict`, optional
			Dictionary whose keys are strings denoting the model variables
			on which boundary conditions should be enforced, and whose
			values are dictionaries specifying fundamental properties (units)
			of those fields. If not specified, boundary conditions are
			enforced on all model variables included in the model state.
		"""
		rfps = {
			name: self.reference_state[name].attrs['units']
			for name in self.reference_state if name != 'time'
		}
		fps = rfps if field_properties is None else \
			{key : val for key, val in field_properties.items() if key in rfps}

		fns = tuple(
			name for name in state if name != 'time' and name in fps
		)

		for field_name in fns:
			field_units = fps[field_name].get('units', rfps[field_name]['units'])
			self.enforce_field(state[field_name], field_name, field_units)

	def enforce(self, state, field_names=None):
		"""
		Enforce the horizontal boundary conditions on the passed state,
		which is modified in-place.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		field_names : `tuple`, optional
			Tuple of strings denoting the model variables on which
			boundary conditions should be enforced. If not specified,
			boundary conditions are enforced on all model variables
			included in the model state.
		"""
		fns = \
			tuple(name for name in self.reference_state if name != 'time') \
			if field_names is None else \
			tuple(name for name in field_names if name in self.reference_state)

		fns = tuple(name for name in state if name in fns)

		for field_name in fns:
			try:
				field_units = state[field_name].attrs['units']
			except KeyError:
				raise KeyError("Field {} misses units attribute.".format(field_name))
			self.enforce_field(state[field_name].values, field_name, field_units)

	@staticmethod
	def factory(boundary_type, nx, ny, nb, **kwargs):
		"""
		Parameters
		----------
		boundary_type : str
			The boundary type, identifying the child class to instantiate.
		nx : int
			Number of points featured by the *physical* grid
			along the first horizontal dimension.
		ny : int
			Number of points featured by the *physical* grid
			along the second horizontal dimension.
		nb : int
			Number of boundary layers.
		kwargs :
			Keyword arguments to be directly forwarded to the
			constructor of the child class.

		Returns
		-------
		obj :
			An object of the suitable child class.
		"""
		args = (nx, ny, nb)

		import tasmania.python.grids._horizontal_boundary as module

		if boundary_type == 'relaxed':
			if ny == 1:
				return module.Relaxed1DX(*args, **kwargs)
			if nx == 1:
				return module.Relaxed1DY(*args, **kwargs)
			return module.Relaxed(*args, **kwargs)

		if boundary_type == 'periodic':
			if ny == 1:
				return module.Periodic1DX(*args, **kwargs)
			if nx == 1:
				return module.Periodic1DY(*args, **kwargs)
			return module.Periodic(*args, **kwargs)

