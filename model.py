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
import copy
from datetime import timedelta
import numpy as np

from tasmania.parameterizations.adjustments import AdjustmentMicrophysics
from tasmania.parameterizations.fast_tendencies import FastTendencyMicrophysics
from tasmania.parameterizations.slow_tendencies import SlowTendencyMicrophysics
from tasmania.storages.grid_data import GridData
import tasmania.utils.utils as utils

class Model:
	"""
	This class is intended to represent and run a generic climate or meteorological numerical model.
	A model is made up of:

	* a dynamical core (mandatory);
	* a set of parameterizations providing *slow-varying tendencies*, i.e., physical parameterization	\
		schemes which, within a timestep, are evaluated *before* the dynamical core; they are intended 	\
		to supply the dynamical core with physical tendencies;
	* a set of parameterizations providing *fast-varying tendencies*, i.e., physical parameterization	\
		schemes which, within a timestep, are evaluated *within* the dynamical core; they are intended 	\
		to supply the dynamical core with physical tendencies;
	* a set of *adjustment-performing* parameterizations, i.e., physical parametrization schemes which,		\
		within a timestep, are performed *after* the dynamical core; they are intended to perform physical	\
		adjustements on the state variables and possibly provide some diagnostics.
	"""
	def __init__(self):
		"""
		Default constructor.
		"""
		# Initialize the dycore, the list of tendencies, and the list of adjustments
		self._dycore      		   = None
		self._slow_tendency_params = []
		self._fast_tendency_params = []
		self._adjustment_params    = []

	def set_dynamical_core(self, dycore):
		"""
		Set the dynamical core.

		Parameters
		----------
		dycore : obj
			Instance of a derived class of :class:`~tasmania.dycore.dycore.DynamicalCore` 
			representing the dynamical core.
		"""
		self._dycore = dycore

		# Update the parameterizations
		for slow_tendency_param in self._slow_tendency_params:
			slow_tendency_param.time_levels = dycore.time_levels
		for adjustment_param in self._adjustment_params:
			adjustment_param.time_levels = dycore.time_levels

		# Update the dycore by setting the microphysics scheme
		for slow_tendency_param in self._slow_tendency_params:
			if isinstance(slow_tendency_param, SlowTendencyMicrophysics) and not done:
				self._dycore.microphysics = slow_tendency_param
		for fast_tendency_param in self._fast_tendency_params:
			if isinstance(fast_tendency_param, FastTendencyMicrophysics) and not done:
				self._dycore.microphysics = fast_tendency_param
		for adjustment_param in self._adjustment_params:
			if isinstance(adjustment_param, AdjustmentMicrophysics) and not done:
				self._dycore.microphysics = adjustment_param

		# Update the dycore by setting the list of parameterizations providing fast-varying tendencies
		self._dycore.fast_tendency_parameterizations = self._fast_tendency_params

	def add_slow_tendency_parameterization(self, tendency):
		"""
		Add to the model a parameterization providing slow-varying tendencies.

		Parameters
		----------
		tendency : obj
			Instance of a derived class of :class:`~tasmania.parameterizations.slow_tendencies.SlowTendency` 
			representing a parameterization providing slow-varying tendencies.

		Note
		----
		In a simulation, parameterizations calculating slow-varying tendencies will be executed in the same 
		order they have been added to the model.
		"""
		self._slow_tendency_params.append(tendency)
		
		#
		# Set dependencies
		#
		if self._dycore is not None:
			tendency.time_levels = self._dycore.time_levels

		if isinstance(tendency, SlowTendencyMicrophysics) and self._dycore is not None:
			self._dycore.microphysics = tendency

	def add_fast_tendency_parameterization(self, tendency):
		"""
		Add to the model a parameterization providing fast-varying tendencies.

		Parameters
		----------
		tendency : obj
			Instance of a derived class of :class:`~tasmania.parameterizations.fast_tendencies.FastTendency` 
			representing a parameterization providing fast-varying tendencies.

		Note
		----
		In a simulation, parameterizations calculating fast-varying tendencies will be executed in the same 
		order they have been added to the model.
		"""
		self._fast_tendency_params.append(tendency)
		
		#
		# Set dependencies
		#
		if isinstance(tendency, FastTendencyMicrophysics) and self._dycore is not None:
			self._dycore.microphysics = tendency

		if self._dycore is not None:
			self._dycore.fast_tendency_parameterizations = self._fast_tendency_params

	def add_adjustment_parameterization(self, adjustment):
		"""
		Add an *adjustment-performing* parameterization to the model.

		Parameters
		----------
		adjustment : obj
			Instance of a derived class of :class:`~tasmania.parameterizations.adjustments.Adjustment` 
			representing an adjustment-performing parameterization.

		Note
		----
		In a simulation, adjustment-performing parameterizations will be executed in the same order 
		they have been added to the model.
		"""
		self._adjustment_params.append(adjustment)

		#
		# Set dependencies
		#
		if self._dycore is not None:
			adjustment.time_levels = self._dycore.time_levels 

		if isinstance(adjustment, AdjustmentMicrophysics) and self._dycore is not None:
			self._dycore.microphysics = adjustment

	def __call__(self, dt, simulation_time, state, save_iterations = []):
		"""
		Call operator integrating the model forward in time.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		simulation_time : obj
			:class:`datetime.timedelta` representing the simulation time.
		state : obj
			The initial state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.
		save_freq : `tuple`, optional
			The iterations at which the state should be saved. Default is empty, meaning that only the initial and 
			final states are saved.

		Returns
		-------
		state_out : obj
			The final state, of the same class of :data:`state`.
		state_save : obj
			The sequence of saved states, of the same class of :data:`state`.
		diagnostics_save : obj
		"""
		# Initialize the control variables and copy the timestep
		steps 		 = 0
		first_save   = 1
		elapsed_time = timedelta()
		dt_          = copy.deepcopy(dt)

		# Initialize storages collecting tendencies and diagnostics
		time        = state.time
		tendencies  = GridData(time, state.grid)
		diagnostics = GridData(time, state.grid)

		# Initialize the outputs
		state_out  = copy.deepcopy(state)
		state_save = copy.deepcopy(state)

		#
		# First time step, performed out of the loop to avoid if-statements
		#
		# Update control variables
		steps += 1
		elapsed_time += dt_
		if elapsed_time > simulation_time:
			dt_ = simulation_time - (elapsed_time - dt_)
			elapsed_time = simulation_time

		# Update the time-dependent topography (useful for stability purposes)
		self._dycore.update_topography(elapsed_time)

		# Sequentially perform tendency-providing schemes, and collect tendencies and diagnostics
		for slow_tendency_param in self._slow_tendency_params: 
			tendencies_, diagnostics_ = slow_tendency_param(dt_, state_out)
			tendencies += tendencies_
			diagnostics.extend(diagnostics_)

		# Run the dynamical core, then update the state and the diagnostics
		state_out_, diagnostics_ = self._dycore(dt_, state_out, tendencies, diagnostics)
		state_out.extend(state_out_)
		diagnostics.extend(diagnostics_)

		# Run the adjustment-performing parameterizations; after each parameterization, 
		# update the state and collect diagnostics
		for adjustment_param in self._adjustment_params:
			diagnostics_, state_out_ = adjustment_param(state_out, dt_)
			state_out.update(state_out_)
			diagnostics.extend(diagnostics_)

		# Check the CFL condition
		u_max, u_min = state_out.get_max('x_velocity'), state_out.get_min('x_velocity')
		v_max, v_min = state_out.get_max('y_velocity'), state_out.get_min('y_velocity')
		cfl = state_out.get_cfl(dt_)
		print('Step %6.i, CFL number: %5.5E, u max: %4.4f m/s, u min: %4.4f m/s, v max: %4.4f m/s, v min %4.4f m/s' 
			  % (steps, cfl, u_max, u_min, v_max, v_min))

		# Save, if needed
		if (steps in save_iterations) or (elapsed_time == simulation_time):
			state_save.grid.update_topography(elapsed_time)
			state_save.append(state_out)

			diagnostics_save = copy.deepcopy(diagnostics)
			diagnostics_save.grid.update_topography(elapsed_time)

			first_save = 0

			print('Step %6.i saved' % (steps))

		#
		# Time loop
		#
		while elapsed_time < simulation_time:
			# Update control variables
			steps += 1
			elapsed_time += dt_
			if elapsed_time > simulation_time:
				dt_ = simulation_time - (elapsed_time - dt_)
				elapsed_time = simulation_time

			# Update the time-dependent topography (useful for stability purposes)
			self._dycore.update_topography(elapsed_time)

			# Sequentially perform tendency-providing schemes, and collect tendencies and diagnostics
			for slow_tendency_param in self._slow_tendency_params: 
				tendencies_, diagnostics_ = slow_tendency_param(dt_, state_out)
				tendencies += tendencies_
				diagnostics.update(diagnostics_)

			# Run the dynamical core, then update the state and the diagnostics
			state_out_, diagnostics_ = self._dycore(dt_, state_out, tendencies, diagnostics)
			state_out.update(state_out_)
			diagnostics.update(diagnostics_)

			# Run the adjustment-performing parameterizations; after each parameterization, 
			# update the state and collect diagnostics
			for adjustment_param in self._adjustment_params:
				diagnostics_, state_out_ = adjustment_param(state_out, dt_)
				state_out.update(state_out_)
				diagnostics.update(diagnostics_)

			# Check the CFL condition
			u_max, u_min = state_out.get_max('x_velocity'), state_out.get_min('x_velocity')
			v_max, v_min = state_out.get_max('y_velocity'), state_out.get_min('y_velocity')
			cfl = state_out.get_cfl(dt_)
			print('Step %6.i, CFL number: %5.5E, u max: %4.4f m/s, u min: %4.4f m/s, v max: %4.4f m/s, v min %4.4f m/s' 
				  % (steps, cfl, u_max, u_min, v_max, v_min))

			# Save, if needed
			if (steps in save_iterations) or (elapsed_time == simulation_time):
				state_save.grid.update_topography(elapsed_time)
				state_save.append(state_out)

				if first_save: 
					diagnostics_save = copy.deepcopy(diagnostics)
					diagnostics_save.grid.update_topography(elapsed_time)
					first_save = 0
				else:
					diagnostics_save.grid.update_topography(elapsed_time)
					diagnostics_save.append(diagnostics)

				print('Step %6.i saved' % (steps))

		return state_out, state_save, diagnostics_save



