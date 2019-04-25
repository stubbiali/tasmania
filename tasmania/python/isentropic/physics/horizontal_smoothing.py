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
	IsentropicHorizontalSmoothing(DiagnosticComponent)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.base_components import DiagnosticComponent


mfwv  = 'mass_fraction_of_water_vapor_in_air'
mfclw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw  = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicHorizontalSmoothing(DiagnosticComponent):
	"""
	Apply numerical smoothing to the prognostic fields of an
	isentropic model state. The class is always instantiated
	over the numerical grid of the underlying domain.
	"""
	def __init__(
		self, domain, smooth_type, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
		moist=False, smooth_moist_coeff=None, smooth_moist_coeff_max=None,
		smooth_moist_damp_depth=None, backend=gt.mode.NUMPY, dtype=np.float64,
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		smooth_type : str
			The type of numerical smoothing to implement.
			See :class:`~tasmania.HorizontalSmoothing` for all available options.
		smooth_coeff : float
			The smoothing coefficient.
		smooth_coeff_max : float
			The maximum value assumed by the smoothing coefficient close to the
			upper boundary.
		smooth_damp_depth : int
			Depth of the damping region.
		moist : `bool`, optional
			:obj:`True` if water species are included in the model and should
			be smoothed, :obj:`False` otherwise. Defaults to :obj:`False`.
		smooth_moist_coeff : `float`, optional
			The smoothing coefficient for the water species.
		smooth_moist_coeff_max : `float`, optional
			The maximum value assumed by the smoothing coefficient for the water
			species close to the upper boundary.
		smooth_damp_depth : int
			Depth of the damping region for the water species.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		self._moist = moist and smooth_moist_coeff is not None

		super().__init__(domain, 'numerical')

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		nb = self.horizontal_boundary.nb

		self._core = HorizontalSmoothing.factory(
			smooth_type, (nx, ny, nz), smooth_coeff, smooth_coeff_max,
			smooth_damp_depth, nb, backend, dtype
		)

		if self._moist:
			smooth_moist_coeff_max = smooth_moist_coeff if smooth_moist_coeff_max is None \
				else smooth_moist_coeff_max
			smooth_moist_damp_depth = 0 if smooth_moist_damp_depth is None \
				else smooth_moist_damp_depth

			self._core_moist = HorizontalSmoothing.factory(
				smooth_type, (nx, ny, nz), smooth_moist_coeff, smooth_moist_coeff_max,
				smooth_moist_damp_depth, nb, backend, dtype
			)
		else:
			self._core_moist = None

		self._s_out  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_out = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_out = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._qv_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qc_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qr_out = np.zeros((nx, ny, nz), dtype=dtype)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._moist:
			return_dict[mfwv]  = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfclw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw]  = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		return self.input_properties

	def array_call(self, state):
		self._core(state['air_isentropic_density'], self._s_out)
		self._core(state['x_momentum_isentropic'],  self._su_out)
		self._core(state['y_momentum_isentropic'],  self._sv_out)
		return_dict = {
			'air_isentropic_density': self._s_out,
			'x_momentum_isentropic':  self._su_out,
			'y_momentum_isentropic':  self._sv_out,
		}

		if self._moist:
			self._core_moist(state[mfwv],  self._qv_out)
			self._core_moist(state[mfclw], self._qc_out)
			self._core_moist(state[mfpw],  self._qr_out)
			return_dict[mfwv]  = self._qv_out
			return_dict[mfclw] = self._qc_out
			return_dict[mfpw]  = self._qr_out

		return return_dict
