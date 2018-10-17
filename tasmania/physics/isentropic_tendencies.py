"""
This module contains:
	NonconservativeIsentropicPressureGradient
	ConservativeIsentropicPressureGradient
	IsentropicVerticalFlux
"""
import numpy as np
from sympl import DataArray, TendencyComponent

import gridtools as gt
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class NonconservativeIsentropicPressureGradient(TendencyComponent):
	"""
	This class calculates the anti-gradient of the Montgomery potential,
	which provides tendencies for the :math:`x`- and :math:`y`-velocity
	in the isentropic system.
	"""
	def __init__(self, grid, order, horizontal_boundary_type,
				 backend=gt.mode.NUMPY, dtype=datatype):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		order : int
			The order of the finite difference formula used to
			discretized the gradient of the Montgomery potential. Either:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Keep track of input parameters
		self._grid     = grid
		self._order	   = order
		self._backend  = backend
		self._dtype	   = dtype

		# Call parent's constructor
		super().__init__()

		# Instantiate the class taking care of the lateral boundary conditions
		self._hboundary = HorizontalBoundary.factory(horizontal_boundary_type,
													 grid, self.nb)

		# Initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery
		# potential; it will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'x_velocity':
				{'dims': dims, 'units': 'm s^-2'},
			'y_velocity':
				{'dims': dims, 'units': 'm s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		"""
		Returns
		-------
		int :
			Number of halo layers in the horizontal directions.
		"""
		if self._order == 2:
			return 1
		elif self._order == 4:
			return 2
		else:
			import warnings
			warnings.warn('Order {} not supported; set order to 2.'.format(self._order))
			self._order = 2
			return 1

	@property
	def _stencil_defs(self):
		if self._order == 2:
			return self._stencil_second_order_defs
		elif self._order == 4:
			return self._stencil_fourth_order_defs

	def array_call(self, state):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Instantiate the GT4Py stencil calculating the pressure gradient
		if self._stencil is None:
			self._stencil_initialize()

		# Extend/shrink the Montgomery potential to accommodate for the
		# lateral boundary conditions
		mtg = state['montgomery_potential']
		self._in_mtg[...] = self._hboundary.from_physical_to_computational_domain(mtg)

		# Run the stencil
		self._stencil.compute()

		# Bring the stencil's outputs back to the physical domain shape.
		# Note that we do not enforce the boundary conditions on the
		# Montgomery potential. Therefore, in the case of relaxed boundary
		# conditions, the outermost halo layers in the output fields would
		# be zero. Nevertheless, periodic conditions are applied exactly.
		u_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_u_tnd, (nx, ny, nz))
		v_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_v_tnd, (nx, ny, nz))

		# Set the other return dictionary
		tendencies = {'x_velocity': u_tnd, 'y_velocity': v_tnd}

		return tendencies, {}

	def _stencil_initialize(self):
		# Shortcuts
		mi, mj, mk = self._hboundary.mi, self._hboundary.mj, self._grid.nz
		nb = self.nb

		# Allocate the NumPy arrays which serve as stencil's input
		self._in_mtg = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Allocate the NumPy arrays which serve as stencil's output
		self._out_u_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._out_v_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_mtg': self._in_mtg},
			outputs			 = {'out_u_tnd': self._out_u_tnd,
								   'out_v_tnd': self._out_v_tnd},
			domain			 = gt.domain.Rectangle((nb, nb, 0),
													 (mi-nb-1, mj-nb-1, mk-1)),
			mode			 = self._backend
		)

	def _stencil_second_order_defs(self, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# Define the computations
		out_u_tnd[i, j] = (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_v_tnd[i, j] = (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_u_tnd, out_v_tnd

	def _stencil_fourth_order_defs(self, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# Define the computations
		out_u_tnd[i, j] = (- in_mtg[i-2, j] + 8. * in_mtg[i-1, j]
						   - 8. * in_mtg[i+1, j] + in_mtg[i+2, j]) / (12. * dx)
		out_v_tnd[i, j] = (- in_mtg[i, j-2] + 8. * in_mtg[i, j-1]
						   - 8. * in_mtg[i, j+1] + in_mtg[i, j+2]) / (12. * dy)

		return out_u_tnd, out_v_tnd


class ConservativeIsentropicPressureGradient(TendencyComponent):
	"""
	This class calculates the anti-gradient of the Montgomery potential,
	multiplied by the air isentropic density. This quantity provides
	tendencies for the :math:`x`- and :math:`y`-momentum in the
	isentropic system.
	"""
	def __init__(self, grid, order, horizontal_boundary_type,
				 backend=gt.mode.NUMPY, dtype=datatype):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		order : int
			The order of the finite difference formula used to
			discretized the gradient of the Montgomery potential. Either:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Keep track of input parameters
		self._grid     = grid
		self._order	   = order
		self._backend  = backend
		self._dtype	   = dtype

		# Call parent's constructor
		super().__init__()

		# Instantiate the class taking care of the lateral boundary conditions
		self._hboundary = HorizontalBoundary.factory(horizontal_boundary_type,
													 grid, self.nb)

		# Initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery
		# potential; it will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density':
				{'dims': dims, 'units': 'kg m^-2 K^-1'},
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'x_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		"""
		Returns
		-------
		int :
			Number of halo layers in the horizontal directions.
		"""
		if self._order == 2:
			return 1
		elif self._order == 4:
			return 2
		else:
			import warnings
			warnings.warn('Order {} not supported; set order to 2.'.format(self._order))
			self._order = 2
			return 1

	@property
	def _stencil_defs(self):
		if self._order == 2:
			return self._stencil_second_order_defs
		elif self._order == 4:
			return self._stencil_fourth_order_defs

	def array_call(self, state):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Instantiate the GT4Py stencil calculating the pressure gradient
		if self._stencil is None:
			self._stencil_initialize()

		# Extract the isentropic density and the Montgomery potential from
		# the input state
		s   = state['air_isentropic_density']
		mtg = state['montgomery_potential']

		# Extend/shrink the isentropic density and the Montgomery potential
		# to accommodate for the lateral boundary conditions
		self._in_s[...]   = self._hboundary.from_physical_to_computational_domain(s)
		self._in_mtg[...] = self._hboundary.from_physical_to_computational_domain(mtg)

		# Run the stencil
		self._stencil.compute()

		# Bring the stencil's outputs back to the physical domain shape.
		# Note that we do not enforce the boundary conditions on the
		# Montgomery potential. Therefore, in the case of relaxed boundary
		# conditions, the outermost halo layers in the output fields would
		# be zero. Nevertheless, periodic conditions are applied exactly.
		su_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_su_tnd, (nx, ny, nz))
		sv_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_sv_tnd, (nx, ny, nz))

		# Set the other return dictionary
		tendencies = {'x_momentum_isentropic': su_tnd,
					  'y_momentum_isentropic': sv_tnd}

		return tendencies, {}

	def _stencil_initialize(self):
		# Shortcuts
		mi, mj, mk = self._hboundary.mi, self._hboundary.mj, self._grid.nz
		nb = self.nb

		# Allocate the NumPy arrays which serve as stencil's input
		self._in_s   = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._in_mtg = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Allocate the NumPy arrays which serve as stencil's output
		self._out_su_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._out_sv_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_s': self._in_s, 'in_mtg': self._in_mtg},
			outputs			 = {'out_su_tnd': self._out_su_tnd,
								'out_sv_tnd': self._out_sv_tnd},
			domain			 = gt.domain.Rectangle((nb, nb, 0),
												   (mi-nb-1, mj-nb-1, mk-1)),
			mode			 = self._backend
		)

	def _stencil_second_order_defs(self, in_s, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# Define the computations
		out_su_tnd[i, j] = in_s[i, j] * (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_sv_tnd[i, j] = in_s[i, j] * (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_su_tnd, out_sv_tnd

	def _stencil_fourth_order_defs(self, in_s, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# Define the computations
		out_su_tnd[i, j] = in_s[i, j] * (- in_mtg[i-2, j]
										 + 8. * in_mtg[i-1, j]
										 - 8. * in_mtg[i+1, j]
										 + in_mtg[i+2, j]) / (12. * dx)
		out_sv_tnd[i, j] = in_s[i, j] * (- in_mtg[i, j-2]
										 + 8. * in_mtg[i, j-1]
										 - 8. * in_mtg[i, j+1]
										 + in_mtg[i, j+2]) / (12. * dy)

		return out_su_tnd, out_sv_tnd


class IsentropicVerticalFlux(TendencyComponent):
	"""
	This class inherits :class:`sympl.TendencyComponent` to calculate
	the conservative vertical advection flux in isentropic coordinates
	for any prognostic variable included in the isentropic system.
	"""
	def __init__(self, grid, moist_on=False, backend=gt.mode.NUMPY):
		"""
		The constructor.

		Parameters
		----------
		grid : obj
			TODO
		moist_on : `bool`, optional
			TODO
		backend : `obj`, optional
			TODO
		"""
		# Keep track of input arguments
		self._grid     = grid
		self._moist_on = moist_on
		self._backend  = backend

		# Call parent's constructor
		super().__init__()

		# Initialize the pointer to the underlying GT4Py stencil;
		# this will be properly redirected the first time the call
		# operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'tendency_of_air_potential_temperature': {'dims': dims, 'units': 'K s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}
		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}
		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		return 1

	def array_call(self, state):
		# Shortcuts
		dtype = state['air_isentropic_density'].values.dtype

		# Instantiate the stencil object
		if self._stencil is None:
			self._stencil_initialize(dtype)

		# Set the stencil's inputs
		self._stencil_set_inputs(state)

		# Run the stencil
		self._stencil.compute()

		# Collect the output arrays in a dictionary
		tendencies = {
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist_on:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv
			tendencies['mass_fraction_of_cloud_liquid_water_in_air'] = self._out_qc
			tendencies['mass_fraction_of_precipitation_water_in_air'] = self._out_qr

		return tendencies, {}

	def _stencil_initialize(self, dtype):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self.nb

		# Allocate arrays serving as stencil's inputs
		self._in_theta = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_s     = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su    = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv    = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist_on:
			self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate arrays serving as stencil's outputs
		self._out_s  = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist_on:
			self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Set stencil's inputs
		inputs = {
			'in_theta': self._in_theta, 'in_s': self._in_s,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		if self._moist_on:
			inputs['in_qv'] = self._in_qv
			inputs['in_qc'] = self._in_qc
			inputs['in_qr'] = self._in_qr

		# Set stencil's outputs
		outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv,
		}
		if self._moist_on:
			outputs['out_qv'] = self._out_qv
			outputs['out_qc'] = self._out_qc
			outputs['out_qr'] = self._out_qr

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs 			 = inputs,
			outputs 		 = outputs,
			domain 			 = gt.domain.Rectangle((0, 0, nb), (nx-1, ny-1, nz-nb-1)),
			backend 		 = self._backend
		)

	def _stencil_set_inputs(self, state):
		self._in_theta[...] = state['tendency_of_air_potential_temperature'][...]
		self._in_s[...] 	= state['air_isentropic_density'][...]
		self._in_su[...]    = state['x_momentum_isentropic'][...]
		self._in_sv[...]    = state['y_momentum_isentropic'][...]
		if self._moist_on:
			self._in_qv[...] = state['mass_fraction_of_water_vapor_in_air'][...]
			self._in_qc[...] = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
			self._in_qr[...] = state['mass_fraction_of_precipitation_water_in_air'][...]

	def _stencil_defs(self, in_theta, in_s, in_su, in_sv,
					  in_qv=None, in_qc=None, in_qr=None):
		# Shortcuts
		dz = self._grid.dz.to_units('K').values.item()

		# Indices
		k = gt.Index(axis=2)

		# Output fields
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_qv = gt.Equation()
			out_qc = gt.Equation()
			out_qr = gt.Equation()

		# Computations
		out_s[k]  = 0.5 * (in_theta[k-1] * in_s[k-1] - in_theta[k+1] * in_s[k+1]) / dz
		out_su[k] = 0.5 * (in_theta[k-1] * in_su[k-1] - in_theta[k+1] * in_su[k+1]) / dz
		out_sv[k] = 0.5 * (in_theta[k-1] * in_sv[k-1] - in_theta[k+1] * in_sv[k+1]) / dz
		if self._moist_on:
			out_qv[k] = 0.5 * (in_theta[k-1] * in_qv[k-1] - in_theta[k+1] * in_qv[k+1]) / dz
			out_qc[k] = 0.5 * (in_theta[k-1] * in_qc[k-1] - in_theta[k+1] * in_qc[k+1]) / dz
			out_qr[k] = 0.5 * (in_theta[k-1] * in_qr[k-1] - in_theta[k+1] * in_qr[k+1]) / dz

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_qv, out_qc, out_qr
