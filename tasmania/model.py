import copy

from sympl import InvalidPropertyDictError
from tasmania.utils.utils import check_property_compatibility


class Model:
	"""
	This class is intended to represent and simulate a generic
	weather or climate numerical model. A model always include
	a dynamical core, and may contain physical parameterizations.
	Within a timestep, the latter could be evaluated either
	*before* or *after* the dynamics.
	"""
	def __init__(self, dycore, physics_before_dynamics=None,
				 physics_after_dynamics=None):
		"""
		The constructor.

		Parameters
		----------
		dycore : obj
			Instance of a derived class of :class:`~tasmania.DynamicalCore`
			representing the dynamical core.
		physics_before_dynamics : `obj`, optional
			Instance of :class:`~tasmania.PhysicsCompositeComponent`
			wrapping the physical parameterizations to evaluate before
			the dynamics is resolved.
		physics_after_dynamics : `obj`, optional
			Instance of :class:`~tasmania.PhysicsCompositeComponent`
			wrapping the physical parameterizations to evaluate after
			the dynamics is resolved.
		"""
		check_components_coherency(dycore, physics_before_dynamics,
								   physics_after_dynamics)

		self._dycore = dycore
		self._physics_before_dynamics = physics_before_dynamics
		self._physics_after_dynamics = physics_after_dynamics

		self._initial_time = None
		self._tendencies   = {}

	@property
	def input_properties(self):
		return_dict = {}

		if self._physics_before_dynamics is None:
			return_dict.update(self._dycore.input_properties)
		else:
			return_dict.update(self._physics_before_dynamics.input_properties)

			dyn_inputs_available = self._physics_before_dynamics.output_properties
			dyn_inputs_required  = self._dycore.input_properties
			for key, value in dyn_inputs_required.items():
				if key not in dyn_inputs_available:
					return_dict[key] = {}
					return_dict[key].update(value)

		return return_dict

	@property
	def output_properties(self):
		return_dict = {}

		if self._physics_after_dynamics is None:
			return_dict.update(self._dycore.output_properties)
		else:
			return_dict.update(self._physics_after_dynamics.output_properties)

			for key, val in self._dycore.output_properties.items():
				if key not in self._physics_after_dynamics.output_properties:
					return_dict[key] = {}
					return_dict[key].update(val)

		return return_dict

	def __call__(self, state, timestep, initial_time=None):
		"""
		Perform a full timestep.

		Parameters
		----------
		state : dictionary
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`~sympl.DataArray`\s
			storing the current values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
		initial_time : datetime
			:class:`datetime.datetime` representing the starting simulation
			time. This is used to update the (time-dependent) topography.
			If not specified, the starting time is inferred from the first
			state passed to this object.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`~sympl.DataArray`\s
			storing the new values for those variables.

		Note
		----
		The passed state may be modified in-place if any adjustment-like
		parameterization is performed before the dynamical core is called.
		"""
		# If this is the first timestep: store the starting time
		if initial_time is not None:
			self._initial_time = copy.deepcopy(initial_time)
		elif self._initial_time is None:
			self._initial_time = copy.deepcopy(state['time'])

		# Update the time-dependent topography
		self._dycore.update_topography(state['time'] - self._initial_time + timestep)

		# Physics before dynamics
		if self._physics_before_dynamics is not None:
			tendencies = self._physics_before_dynamics(state, timestep)

			for key, value in self._tendencies.items():
				if key != 'time':
					if key in tendencies:
						tendencies[key] += value
					else:
						tendencies[key] = value
		else:
			tendencies = {}
			tendencies.update(self._tendencies)

		# Dynamics
		state_new = self._dycore(state, tendencies, timestep)

		# Physics after dynamics
		if self._physics_after_dynamics:
			self._tendencies.update(self._physics_after_dynamics(state_new, timestep))

		return state_new


def check_components_coherency(dycore, physics_before_dynamics,
							   physics_after_dynamics):
	tendencies_allowed = dycore.tendency_properties

	if physics_before_dynamics is not None:
		tendencies_available_before = physics_before_dynamics.tendency_properties
		for key in tendencies_available_before:
			if key in tendencies_allowed:
				try:
					check_property_compatibility(tendencies_allowed[key],
											 	 tendencies_available_before[key], name=key)
				except InvalidPropertyDictError as err:
					raise InvalidPropertyDictError(
						'While assessing compatibility between the tendencies output '
						'by the parameterizations run before resolving the dynamics, '
						'and those required by the dynamical core: {}'.format(str(err)))
			else:
				raise KeyError('Tendency {} calculated by the parameterizations, '
							   'but not required by the dynamical core.'.format(key))

		if physics_after_dynamics is not None:
			tendencies_available_after = physics_after_dynamics.tendency_properties
			for key in tendencies_available_after:
				if key in tendencies_available_before:
					try:
						check_property_compatibility(tendencies_available_before[key],
												 	 tendencies_available_after[key], name=key)
					except InvalidPropertyDictError as err:
						print('While assessing inter-parameterizations compatibility.')
						raise err

		inputs_required  = dycore.input_properties
		inputs_available = physics_before_dynamics.output_properties
		for key in inputs_required:
			if key in inputs_available:
				try:
					check_property_compatibility(inputs_required[key],
											 	 inputs_available[key], name=key)
				except InvalidPropertyDictError as err:
					raise InvalidPropertyDictError(
						'While assessing compatibility between the model variables output '
						'by the parameterizations run before resolving the dynamics, '
						'and those required by the dynamical core: {}'.format(str(err)))

	if physics_after_dynamics is not None:
		tendencies_available_after = physics_after_dynamics.tendency_properties
		for key in tendencies_available_after:
			if key not in tendencies_allowed:
				raise KeyError('Tendency {} calculated by the parameterizations, '
							   'but not required by the dynamical core.'.format(key))

		inputs_required = physics_after_dynamics.input_properties
		inputs_available = dycore.output_properties
		for key in inputs_required:
			if key in inputs_available:
				try:
					check_property_compatibility(inputs_required[key],
											 	 inputs_available[key], name=key)
				except InvalidPropertyDictError as err:
					raise InvalidPropertyDictError(
						'While assessing compatibility between the model variables output '
						'by the dynamical core, and those required by the parameterizations '
						'run after resolving the dynamics: {}'.format(str(err)))
			else:
				raise KeyError('Variable {} required by the parameterizations '
							   'performed after the dynamical core, but not '
							   'provided by the dynamical core.'.format(key))
