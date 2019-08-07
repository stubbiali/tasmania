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
	IsentropicHorizontalDiffusion(TendencyComponent)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.framework.base_components import TendencyComponent


mfwv  = 'mass_fraction_of_water_vapor_in_air'
mfclw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw  = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicHorizontalDiffusion(TendencyComponent):
	"""
	Calculate the tendencies due to horizontal diffusion for the
	prognostic fields of an isentropic model state. The class is
	always instantiated over the numerical grid of the
	underlying domain.
	"""
	def __init__(
		self, domain, diffusion_type, diffusion_coeff, diffusion_coeff_max,
		diffusion_damp_depth, tracers=None, diffusion_tracer_coeff=None,
		diffusion_tracer_coeff_max=None, diffusion_tracer_damp_depth=None,
		backend=gt.mode.NUMPY, dtype=np.float64, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		diffusion_type : str
			The type of numerical diffusion to implement.
			See :class:`~tasmania.HorizontalDiffusion` for all available options.
		diffusion_coeff : sympl.DataArray
			1-item array representing the diffusion coefficient;
			in units compatible with [s^-1].
		diffusion_coeff_max : sympl.DataArray
			1-item array representing the maximum value assumed by the
			diffusion coefficient close to the upper boundary;
			in units compatible with [s^-1].
		diffusion_damp_depth : int
			Depth of the damping region.
		tracers : `dict`, optional
			Dictionary whose keys are strings denoting the tracers included in
			the model, and whose values are	dictionaries specifying fundamental
			properties ('units') for those tracers.
		diffusion_tracer_coeff : `sympl.DataArray`, optional
			1-item array representing the diffusion coefficient for the
			tracers; in units compatible with [s^-1].
		diffusion_tracer_coeff_max : `sympl.DataArray`, optional
			1-item array representing the maximum value assumed by the
			diffusion coefficient for the tracers close to the upper boundary;
			in units compatible with [s^-1].
		diffusion_tracer_damp_depth : int
			Depth of the damping region for the tracers.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Keyword arguments to be directly forwarded to the parent constructor.
		"""
		self._tracers = {} if tracers is None else tracers
		self._moist = len(self._tracers) and diffusion_tracer_coeff is not None

		super().__init__(domain, 'numerical', **kwargs)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()
		nb = self.horizontal_boundary.nb

		diff_coeff = diffusion_coeff.to_units('s^-1').values.item()
		diff_coeff_max = diffusion_coeff_max.to_units('s^-1').values.item()

		self._core = HorizontalDiffusion.factory(
			diffusion_type, (nx, ny, nz), dx, dy,
			diff_coeff, diff_coeff_max, diffusion_damp_depth,
			nb, backend, dtype
		)

		if self._moist:
			diff_tracer_coeff = \
				diffusion_tracer_coeff.to_units('s^-1').values.item()
			diff_tracer_coeff_max = \
				diff_tracer_coeff if diffusion_tracer_coeff_max is None \
				else diffusion_tracer_coeff_max.to_units('s^-1').values.item()
			diff_tracer_damp_depth = \
				0 if diffusion_tracer_damp_depth is None \
				else diffusion_tracer_damp_depth

			self._core_tracer = HorizontalDiffusion.factory(
				diffusion_type, (nx, ny, nz), dx, dy,
				diff_tracer_coeff, diff_tracer_coeff_max, diff_tracer_damp_depth,
				nb, backend, dtype
			)
		else:
			self._core_tracer = None

		self._s_tnd  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._q_tnd  = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._moist:
			return_dict.update({
				tracer: {'dims': dims, 'units': props['units']}
				for tracer, props in self._tracers.items()
			})

		return return_dict

	@property
	def tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		if self._moist:
			return_dict.update({
				tracer: {'dims': dims, 'units': props['units'] + ' s^-1'}
				for tracer, props in self._tracers.items()
			})

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		self._core(state['air_isentropic_density'], self._s_tnd)
		self._core(state['x_momentum_isentropic'],  self._su_tnd)
		self._core(state['y_momentum_isentropic'],  self._sv_tnd)
		return_dict = {
			'air_isentropic_density': self._s_tnd,
			'x_momentum_isentropic':  self._su_tnd,
			'y_momentum_isentropic':  self._sv_tnd,
		}

		if self._moist:
			for tracer in self._tracers:
				self._core_tracer(state[tracer],  self._q_tnd[tracer])
				return_dict[tracer]  = self._q_tnd[tracer]

		return return_dict, {}
