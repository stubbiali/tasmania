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
	BurgersDynamicalCore(DynamicalCore)
"""
import gridtools as gt
from tasmania.conf import datatype
from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.framework.dycore import DynamicalCore


class BurgersDynamicalCore(DynamicalCore):
	"""
	The dynamical core for the inviscid 2-D Burgers equations.
	"""
	def __init__(
		self, domain, intermediate_tendencies=None,
		time_integration_scheme='forward_euler', flux_scheme='upwind',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		intermediate_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the intermediate physical tendencies.
			Here, *intermediate* refers to the fact that these physical
			packages are called *before* each stage of the dynamical core
			to calculate the physical tendencies.
		time_integration_scheme : `str`, optional
			String specifying the time integration scheme to be used.
			Defaults to 'forward_euler'. See :class:`tasmania.BurgersStepper`
			for all available options.
		flux_scheme : `str`, optional
			String specifying the advective flux scheme to be used.
			Defaults to 'upwind'. See :class:`tasmania.BurgersAdvection`
			for all available options.
		backend : `str`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated within
			this class.
		"""
		super().__init__(
			domain, grid_type='computational', time_units='s',
			intermediate_tendencies=intermediate_tendencies,
			intermediate_diagnostics=None, substeps=0,
			fast_tendencies=None, fast_diagnostics=None
		)

		assert self.grid.nz == 1, \
			'The number grid points along the vertical dimension must be 1.'

		self._stepper = BurgersStepper.factory(
			time_integration_scheme, self.grid.grid_xy, self.horizontal_boundary.nb,
			flux_scheme, backend, dtype
		)

	@property
	def _input_properties(self):
		g = self.grid
		dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def _substep_input_properties(self):
		return {}

	@property
	def _tendency_properties(self):
		g = self.grid
		dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-2'},
			'y_velocity': {'dims': dims, 'units': 'm s^-2'},
		}

	@property
	def _substep_tendency_properties(self):
		return {}

	@property
	def _output_properties(self):
		g = self.grid
		dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def _substep_output_properties(self):
		return {}

	@property
	def stages(self):
		return self._stepper.stages

	def substep_fractions(self):
		return 1

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		out_state = self._stepper(stage, raw_state, raw_tendencies, timestep)

		self.horizontal_boundary.enforce_raw(
			out_state,
			field_properties={
				'x_velocity': {'units': 'm s^-1'},
				'y_velocity': {'units': 'm s^-1'},
			}
		)

		return out_state

	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		raise NotImplementedError()
