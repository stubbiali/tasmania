import copy
from datetime import timedelta
import numpy as np

from tasmania.storages.grid_data import GridData
import tasmania.utils.utils as utils

class Model:
	"""
	This class is intended to represent and run a generic climate or meteorological numerical model.
	A model is made up of:

	* a dynamical core (mandatory);
	* a set of *tendency-providing* parameterizations, i.e., physical parameterization schemes which, within a timestep, \
		are performed *before* the dynamical core; they are intended to supply the dynamical core with physical tendencies;
	* a set of *adjustment-performing* parameterizations, i.e., physical parametrization schemes which, within a timestep, \
		are performed *after* the dynamical core; they are intended to perform physical adjustements on the state variables \
		and possibly provide some diagnostics.
	"""
	def __init__(self):
		"""
		Default constructor.
		"""
		# Initialize the dycore, the list of tendencies, and the list of adjustments
		self._dycore      		= None
		self._tendency_params  	= []
		self._adjustment_params = []

	def set_dynamical_core(self, dycore):
		"""
		Set the dynamical core.

		Parameters
		----------
		dycore : obj
			Instance of a derived class of :class:`~tasmania.dycore.dycore.DynamicalCore` representing the dynamical core.
		"""
		self._dycore = dycore

		# Update the parameterizations
		for tendency_param in self._tendency_params:
			tendency_param.time_levels = dycore.time_levels
		for adjustment_param in self._adjustment_params:
			adjustment_param.time_levels = dycore.time_levels

	def add_tendency(self, tendency):
		"""
		Add a *tendency-providing* parameterization to the model.

		Parameters
		----------
		tendency : obj
			Instance of a derived class of :class:`~tasmania.parameterizations.tendency.Tendency` representing a 
			tendency-providing parameterization.

		Note
		----
		In a simulation, tendency-providing parameterizations will be executed in the same order they have been added to the model.
		"""
		self._tendency_params.append(tendency)
		if self._dycore is not None:
			self._tendency_params[-1].time_levels = self._dycore.time_levels

	def add_adjustment(self, adjustment):
		"""
		Add an *adjustment-performing* parameterization to the model.

		Parameters
		----------
		adjustment : obj
			Instance of a derived class of :class:`~tasmania.parameterizations.adjustment.Adjustment` representing an 
			adjustment-performing parameterization.

		Note
		----
		In a simulation, adjustment-performing parameterizations will be executed in the same order they have been added to the model.
		"""
		self._adjustment_params.append(adjustment)
		if self._dycore is not None:
			self._adjustment_params[-1].time_levels = self._dycore.time_levels 

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
		"""
		# Initialize the control variables and copy the timestep
		steps = 0
		elapsed_time = timedelta()
		dt_ = copy.deepcopy(dt)

		# Initialize storages collecting tendencies and diagnostics
		time = utils.convert_datetime64_to_datetime(state[state.variable_names[0]].coords['time'].values[0])
		tendencies = GridData(time, state.grid)
		diagnostics = GridData(time, state.grid)

		# Initialize the outputs
		state_out = copy.deepcopy(state)
		state_save = copy.deepcopy(state)

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
			for tendency_param in self._tendency_params: 
				tendencies_, diagnostics_ = tendency_param(dt_, state_out)
				tendencies.update(tendencies_)
				diagnostics.update(diagnostics_)

			# Run the dynamical core, then update the state
			state_new = self._dycore(dt_, state_out, diagnostics, tendencies)
			state_out.update(state_new)

			# Run the adjustment-performing parameterizations; after each parameterization, update the state and collect diagnostics
			for adjustment_param in self._adjustment_params:
				state_new, diagnostics_ = adjustment_param(dt_, state_out)
				state_out.update(state_new)
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
				print('Step %6.i saved' % (steps))

		return state_out, state_save



