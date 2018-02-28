import copy
from datetime import timedelta
import numpy as np

class Model:
	"""
	This class is intended to represent and run a generic climate or meteorological numerical model.
	"""
	def __init__(self, dynamical_core, diagnostics = []):
		"""
		Constructor.

		Parameters
		----------
		dynamical_core : obj 
			An instance of :class:`~dycore.dycore.DynamicalCore` or one of its derived classes, implementing a dynamical core.
		diagnostics : `list`, optional 
			List of diagnostics. Default is empty.
		"""
		self._dycore = dynamical_core
		self._diagnostics = diagnostics

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

			# Run the dynamical core and update the state
			state_new = self._dycore(dt_, state_out)
			state_out.update(state_new)

			# Run the diagnostics; after each diagnostic, update the state
			for diagnostic in self._diagnostics:
				state_new = diagnostic(state_out)
				state_out.update(state_new)

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



