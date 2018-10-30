"""
This module contain:
	ForwardEuler
	RungeKutta2
	RungeKutta3COSMO
	RungeKutta3
"""
import abc
from sympl import DataArray, TendencyStepper as SymplTendencyStepper
from sympl._components.timesteppers import convert_tendencies_units_for_state
from sympl._core.units import clean_units

from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.utils.data_utils import add, multiply


def get_increment(state, timestep, prognostic):
	# Calculate tendencies and retrieve diagnostics
	tendencies, diagnostics = prognostic(state, timestep)

	# Convert tendencies in units compatible with the state
	convert_tendencies_units_for_state(tendencies, state)

	# Calculate the increment
	increment = multiply(timestep.total_seconds(), tendencies)

	# Set the correct units for the increment of each variable
	for key, val in increment.items():
		if isinstance(val, DataArray) and 'units' in val.attrs.keys():
			val.attrs['units'] += ' s'
			val.attrs['units'] = clean_units(val.attrs['units'])

	return increment, diagnostics


class TendencyStepper(SymplTendencyStepper):
	"""
	As its parent :class:`sympl.TendencyStepper`, this abstract base
	class heads a hierarchy of components which integrate some
	prognostic model variables based on the tendencies computed
	by a set of internal :class:`sympl.TendencyComponent` objects.
	In addition to :class:`sympl.TendencyStepper`, this class is
	equipped with utilities to handle the horizontal boundary
	conditions are provided.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, *args, grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.PrognosticComponent` providing
			tendencies for the prognostic variables.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		super().__init__(*args)

		if horizontal_boundary_type is not None:
			# Determine the number of boundary layers by inspecting
			# the attribute nb of each prognostic component
			nb = 1
			for component in self.prognostic_list:
				nb = max(nb, getattr(component, 'nb', 1))

			# Instantiate the object which will take care of
			# boundary conditions
			self._bnd = HorizontalBoundary.factory(horizontal_boundary_type,
												   grid, nb)
		else:
			self._bnd = None

		self._damp_on = True


class ForwardEuler(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the forward Euler time integration scheme.
	"""
	def __init__(self, *args, grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.PrognosticComponent` providing
			tendencies for the prognostic variables.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		super().__init__(*args, grid=grid,
						 horizontal_boundary_type=horizontal_boundary_type)

	def _call(self, state, timestep):
		# Shortcuts
		dt = timestep.total_seconds()
		out_units = {name: properties['units']
					 for name, properties in self.output_properties.items()}

		# Calculate tendencies and diagnostics
		tendencies, diagnostics = self.prognostic(state, timestep)

		# Convert tendencies in units compatible with the state
		convert_tendencies_units_for_state(tendencies, state)

		# Set the correct units for the increment of each variable
		for key, val in tendencies.items():
			if isinstance(val, DataArray) and 'units' in val.attrs.keys():
				val.attrs['units'] += ' s'
				val.attrs['units'] = clean_units(val.attrs['units'])

		# Step the solution
		out_state = add(state, multiply(dt, tendencies), units=out_units,
					    unshared_variables_in_output=True)

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values,
								  state[name].to_units(out_units[name]).values)

		return diagnostics, out_state


class RungeKutta2(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the two-stages, second-order Runge-Kutta scheme
	as described in the reference.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""
	def __init__(self, *args, grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.PrognosticComponent` providing
			tendencies for the prognostic variables.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		super().__init__(*args, grid=grid,
						 horizontal_boundary_type=horizontal_boundary_type)

	def _call(self, state, timestep):
		# Shortcuts
		out_units = {name: properties['units']
					 for name, properties in self.output_properties.items()}

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 = add(state, multiply(0.5, k0), units=out_units,
					  unshared_variables_in_output=True)
		state_1['time'] = state['time'] + 0.5*timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_1[name].values,
								  state[name].to_units(out_units[name]).values)

		# Second stage
		k1, _ = get_increment(state_1, timestep, self.prognostic)
		out_state = add(state, k1, units=out_units,
						unshared_variables_in_output=True)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_1[name].values)

		return diagnostics, out_state


class RungeKutta3COSMO(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the three-stages Runge-Kutta scheme as used in the
	`COSMO model <http://www.cosmo-model.org>`_. This integrator is
	nominally second-order, and third-order for linear problems.

	References
	----------
	Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
		regional COSMO-model. Part I: Dynamics and numerics.* \
		Deutscher Wetterdienst, Germany.
	"""
	def __init__(self, *args, grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.PrognosticComponent` providing
			tendencies for the prognostic variables.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		super().__init__(*args, grid=grid,
						 horizontal_boundary_type=horizontal_boundary_type)

	def _call(self, state, timestep):
		# Shortcuts
		out_units = {name: properties['units']
					 for name, properties in self.output_properties.items()}

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 = add(state, multiply(1./3., k0), units=out_units,
					  unshared_variables_in_output=True)
		state_1['time'] = state['time'] + 1.0/3.0 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_1[name].values,
								  state[name].to_units(out_units[name]).values)

		# Second stage
		k1, _ = get_increment(state_1, timestep, self.prognostic)
		state_2 = add(state, multiply(1./2., k1), units=out_units,
					  unshared_variables_in_output=True)
		state_2['time'] = state['time'] + 1.0/2.0 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_2[name].values, state_1[name].values)

		# Second stage
		k2, _ = get_increment(state_2, timestep, self.prognostic)
		out_state = add(state, k2, units=out_units,
						unshared_variables_in_output=True)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_2[name].values)

		return diagnostics, out_state


class RungeKutta3(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the three-stages, third-order Runge-Kutta scheme
	as described in the reference.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""
	def __init__(self, *args, grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.PrognosticComponent` providing
			tendencies for the prognostic variables.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		super().__init__(*args, grid=grid,
						 horizontal_boundary_type=horizontal_boundary_type)

		# Free parameters for RK3
		self._alpha1 = 1./2.
		self._alpha2 = 3./4.

		# Set the other parameters yielding a third-order method
		self._gamma1 = (3.*self._alpha2 - 2.) / \
					   (6. * self._alpha1 * (self._alpha2 - self._alpha1))
		self._gamma2 = (3.*self._alpha1 - 2.) / \
					   (6. * self._alpha2 * (self._alpha1 - self._alpha2))
		self._gamma0 = 1. - self._gamma1 - self._gamma2
		self._beta21 = self._alpha2 - 1. / (6. * self._alpha1 * self._gamma2)

	def _call(self, state, timestep):
		# Shortcuts
		out_units  = {name: properties['units']
					  for name, properties in self.output_properties.items()}
		a1, a2     = self._alpha1, self._alpha2
		b21        = self._beta21
		g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 		= add(state, multiply(a1, k0), units=out_units,
					  		  unshared_variables_in_output=True)
		state_1['time'] = state['time'] + a1 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_1[name].values,
								  state[name].to_units(out_units[name]).values)

		# Second stage
		k1, _ 	= get_increment(state_1, timestep, self.prognostic)
		state_2 = add(state,
					  add(multiply(b21, k0), multiply((a2 - b21), k1)),
					  units=out_units, unshared_variables_in_output=True)
		state_2['time'] = state['time'] + a2 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_2[name].values, state_1[name].values)

		# Second stage
		k2, _     = get_increment(state_2, timestep, self.prognostic)
		k1k2      = add(multiply(g1, k1), multiply(g2, k2))
		k0k1k2 	  = add(multiply(g0, k0), k1k2)
		out_state = add(state, k0k1k2, units=out_units,
						unshared_variables_in_output=True)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_2[name].values)

		return diagnostics, out_state
