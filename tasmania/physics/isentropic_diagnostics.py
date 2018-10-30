"""
This module contains:
	IsentropicDiagnostics
	IsentropicVelocityComponents
"""
import numpy as np
from sympl import DataArray, DiagnosticComponent

import gridtools as gt
from tasmania.dynamics.diagnostics import IsentropicDiagnostics as Helper, \
										  HorizontalVelocity
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class IsentropicDiagnostics(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnosed
	the pressure, the Exner function, the Montgomery potential and
	the height of the interface levels. Optionally, the air density
	and temperature are calculated as well.
	"""
	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gravitational_acceleration':
			DataArray(9.80665, attrs={'units': 'm s^-2'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(self, grid, moist_on, pt, backend=gt.mode.NUMPY,
				 dtype=datatype, physical_constants=None, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		moist_on : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils
			implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
        physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.physics.isentropic.IsentropicDiagnostics._d_physical_constants`
			for the default values.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.DiagnosticComponent`.
		"""
		# Keep track of some input parameters
		self._grid, self._moist_on = grid, moist_on
		self._pt = pt.to_units('Pa').values.item()

		# Call parent's constructor
		super().__init__(**kwargs)

		# Instantiate the class computing the diagnostic variables
		self._helper = Helper(grid, backend, dtype, physical_constants)

	@property
	def input_properties(self):
		grid 	  = self._grid
		dims 	  = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density':
				{'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		grid 	  = self._grid
		dims 	  = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
		dims_stgz = (dims[0], dims[1], grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels':
				{'dims': dims_stgz, 'units': 'Pa'},
			'exner_function_on_interface_levels':
				{'dims': dims_stgz, 'units': 'J K^-1 kg^-1'},
			'height_on_interface_levels':
				{'dims': dims_stgz, 'units': 'm'},
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		if self._moist_on:
			return_dict['air_density'] = {'dims': dims, 'units': 'kg m^-3'}
			return_dict['air_temperature'] = {'dims': dims, 'units': 'K'}

		return return_dict

	def array_call(self, state):
		# Extract useful variables from input state dictionary
		s  = state['air_isentropic_density']

		# Calculate all diagnostics, and initialize the output dictionary
		p, exn, mtg, h = self._helper.get_diagnostic_variables(s, self._pt)
		diagnostics = {
			'air_pressure_on_interface_levels': p,
			'exner_function_on_interface_levels': exn,
			'montgomery_potential': mtg,
			'height_on_interface_levels': h,
		}
		if self._moist_on:
			diagnostics['air_density'] = self._helper.get_air_density(s, h)
			diagnostics['air_temperature'] = self._helper.get_air_temperature(exn)

		return diagnostics


class IsentropicVelocityComponents(DiagnosticComponent):
	"""
	This class inherits :class:`sympl.DiagnosticComponent` to retrieve
	the horizontal velocity components with the help of the isentropic
	momenta and the isentropic density.
	"""
	def __init__(self, grid, horizontal_boundary_type, reference_state,
				 backend=gt.mode.NUMPY, dtype=datatype, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		reference_state : dict
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`sympl.DataArray`\s representing
			reference values for those variables. These values may be used
			to enforce the horizontal boundary conditions on the velocity
			components. The dictionary should contain the following variables:

				* 'x_velocity_at_u_locations', in units compatible with [m s^-1];
				* 'y_velocity_at_v_locations', in units compatible with [m s^-1].

		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.DiagnosticComponent`.
		"""
		self._grid = grid

		super().__init__(**kwargs)

		self._helper = HorizontalVelocity(grid, backend, dtype)
		self._hboundary = HorizontalBoundary.factory(horizontal_boundary_type, grid, 1)

		try:
			uref = reference_state['x_velocity_at_u_locations']
			try:
				self._uref = np.copy(uref.to_units('m s^-1').values)
			except ValueError as e:
				print('The field ''x_velocity_at_u_locations'' in the input '
					  'dictionary ''reference_state'' should be expressed in units '
					  'compatible with ''m s^-1''.')
				raise e
		except KeyError as e:
			print('The input dictionary ''reference_state'' should contain the '
				  'field ''x_velocity_at_u_locations''.')
			raise e

		try:
			vref = reference_state['y_velocity_at_v_locations']
			try:
				self._vref = np.copy(vref.to_units('m s^-1').values)
			except ValueError as e:
				print('The field ''y_velocity_at_v_locations'' in the input '
					  'dictionary ''reference_state'' should be expressed in units '
					  'compatible with ''m s^-1''.')
				raise e
		except KeyError as e:
			print('The input dictionary ''reference_state'' should contain the '
				  'field ''y_velocity_at_v_locations''.')
			raise e

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		grid = self._grid
		dims_x = (grid.x_at_u_locations.dims[0], grid.y.dims[0], grid.z.dims[0])
		dims_y = (grid.x.dims[0], grid.y_at_v_locations.dims[0], grid.z.dims[0])

		return_dict = {
			'x_velocity_at_u_locations': {'dims': dims_x, 'units': 'm s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_y, 'units': 'm s^-1'},
		}

		return return_dict

	def array_call(self, state):
		# Extract the required model variables from the input state
		s  = state['air_isentropic_density']
		su = state['x_momentum_isentropic']
		sv = state['y_momentum_isentropic']

		# Diagnose the velocity components
		u, v = self._helper.get_velocity_components(s, su, sv)

		# Enforce the boundary conditions
		self._hboundary.set_outermost_layers_x(u, self._uref)
		self._hboundary.set_outermost_layers_y(v, self._vref)

		# Instantiate the output dictionary
		diagnostics = {'x_velocity_at_u_locations': u, 'y_velocity_at_v_locations': v}

		return diagnostics
