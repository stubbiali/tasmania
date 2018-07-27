from sympl import DiagnosticComponent as Diagnostic, \
				  PrognosticComponent as Tendency, \
				  AdamsBashforth as TendencyStepper, \
				  combine_component_properties

from tasmania.utils.data_utils import add
from tasmania.utils.utils import assert_sequence, check_property_compatibility


class PhysicsComponentComposite:
	"""
	This class automates the sequential execution of a set of
	physical parameterizations.
	"""
	def __init__(self, mode, *args):
		"""
		Constructor.

		Parameters
		----------
		mode : str
			Modality in which physical components should be coupled.
			Available options are:

				* 'ps' or 'parallel_splitting', for parallel splitting (PS);
				* 'sus' or 'sequential_update_splitting', for sequential-update \
					splitting (SUS).

		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`.

		Note
		----
		In SUS mode, each passed :class:`sympl.PrognosticComponent` is wrapped
		within a :class:`sympl.AdamsBashforth` object implementing the forward
		Euler timestepping scheme.

		References
		----------
		Donahue, A. S., and P. M. Caldwell. (2018). \
		Impact of physics parameterization ordering in a global atmosphere model. \
		*Journal of Advances in Modeling earth Systems*, *10*:481-499.

		Raises
		------
		ValueError :
			If an invalid :obj:`mode` is specified.
		"""
		assert_sequence(args, reftype=(Diagnostic, Tendency))

		self._mode = mode.lower()

		if mode in ('ps', 'parallel_splitting'):
			self._initialize_components_list_ps(args)
			self._call = self._call_ps
		elif mode in ('sus', 'sequential_update_splitting'):
			self._initialize_components_list_sus(args)
			self._call = self._call_sus
		else:
			raise ValueError

	@property
	def input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting model variables
			which must be present in any input state, and whose values are
			dictionaries specifying fundamental properties (dims, units)
			for those variables.
		"""
		# Initialize the return dictionary, i.e., the list of requirements
		return_dict = {}

		# Initialize the properties of the variables which the state will be
		# including after having passed it to __call__
		output_properties = {}

		for component in self.components_list:
			# Get the set of variables which should be passed to
			# the component, and which are already at disposal
			already_at_disposal = \
				set(component.input_properties.keys()).intersection(output_properties.keys())

			# Ensure the requirements of the component are compatible
			# with the variables already at disposal
			for name in already_at_disposal:
				check_property_compatibility(output_properties[name],
											 component.input_properties[name],
											 name=name)

			# Check if there exists any variable which the component
			# requires but which is not yet at disposal
			not_at_disposal = \
				set(component.input_properties.keys()).difference(output_properties.keys())

			for name in not_at_disposal:
				# Add the missing variable to the requirements and
				# to the output state
				return_dict[name] = {}
				return_dict[name].update(component.input_properties[name])
				output_properties[name] = {}
				output_properties[name].update(component.input_properties[name])

			# Update the properties of the variables which the state
			# will be including after having passed it to __call__
			for name, properties in component.diagnostic_properties.items():
				if name not in output_properties.keys():
					output_properties[name] = {}
				output_properties[name].update(properties)

		return return_dict

	@property
	def output_properties(self):
		"""
		Return
		------
		dict :
            Dictionary whose keys are strings denoting model variables
            which will be present in the input state when the call operator
            will return, and whose values are dictionaries specifying
            fundamental properties (dims, units) for those variables.
		"""
		# Initialize the return dictionary
		return_dict = {}

		for component in self.components_list:
			# Get the set of variables which should be passed to
			# the component, and which are already at disposal
			already_at_disposal = \
				set(component.input_properties.keys()).intersection(return_dict.keys())

			# Ensure the requirements of the component are compatible
			# with the variables already at disposal
			for name in already_at_disposal:
				check_property_compatibility(return_dict[name],
											 component.input_properties[name],
											 name=name)

			# Check if there exists any variable which the component
			# requires but which is not yet at disposal
			not_at_disposal = \
				set(component.input_properties.keys()).difference(return_dict.keys())

			for name in not_at_disposal:
				# Add the missing variable to the return dictionary
				return_dict[name] = {}
				return_dict[name].update(component.input_properties[name])

			# Update the return dictionary
			for name, properties in component.diagnostic_properties.items():
				if name not in return_dict.keys():
					return_dict[name] = {}
				return_dict[name].update(properties)

		return return_dict

	@property
	def tendency_properties(self):
		"""
		Return
		------
		dict :
            Dictionary whose keys are strings denoting tendencies
            which are computed by this object, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            for those tendencies.
		"""
		if self._mode in ('ps', 'parallel_splitting'):
			# Only Tendency components calculate tendencies
			tendency_list = [c for c in self.components_list if isinstance(c, Tendency)]
			return combine_component_properties(tendency_list, 'tendency_properties')
		elif self._mode in ('sus', 'sequential_update_splitting'):
			# No tendencies are calculated
			return {}

	def __call__(self, state, timestep):
		"""
		Sequentially invoked all the parameterizations.
		All tendencies calculated by the :class:`sympl.PrognosticComponent`
		objects are summed up, and the input state is modified *in-place*
		by updating it via the diagnostics calculated by the
		:class:`sympl.PrognosticComponent` and :class:`sympl.DiagnosticComponent`
		objects, and the new model variable values output by the
		:class:`sympl.AdamsBashforth` components.

		Parameters
		----------
		state : dict
			The input state as a dictionary whose keys are strings denoting
			the model variables, and whose values are :sympl:`DataArray`\s
			storing values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size, i.e.,
			the amount of time to step forward.

		Return
		------
		dict :
            Dictionary whose keys are strings denoting the output
            tendencies, and whose values are :class:`sympl.DataArray`\s
            storing values for those tendencies.
		"""
		return self._call(state, timestep)

	def _initialize_components_list_ps(self, components):
		"""
		This method initializes the list of physical components
		when the parallel splitting approach has to be pursued.
		"""
		self.components_list = components

	def _initialize_components_list_sus(self, components):
		"""
		This method initializes the list of physical components
		when sequential-update splitting approach has to be pursued.
		"""
		self.components_list = []

		for component in components:
			if isinstance(component, Tendency):
				self.components_list.append(TendencyStepper(component, order=1))
			else:
				self.components_list.append(component)

	def _call_ps(self, state, timestep=None):
		"""
		Couple the parameterizations pursuing the parallel splitting approach.
		"""
		out_tendencies  = {'time': state['time']}

		for component in self.components_list:
			if isinstance(component, Tendency):
				tendencies, diagnostics = component(state)
				out_tendencies = add(out_tendencies, tendencies,
									 units=self.tendency_properties)
				state.update(diagnostics)
			else:
				diagnostics = component(state)
				state.update(diagnostics)

		return out_tendencies

	def _call_sus(self, state, timestep):
		"""
		Couple the parameterizations pursuing the parallel splitting approach.
		"""
		out_tendencies = {'time': state['time']}

		for component in self.components_list:
			if isinstance(component, TendencyStepper):
				diagnostics, new_state = component(state, timestep)
				state.update(diagnostics)
				state.update(new_state)

				# Ensure the state is still defined at the current timestep
				state['time'] = out_tendencies['time']
			else:
				diagnostics = component(state)
				state.update(diagnostics)

		return out_tendencies
