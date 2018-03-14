import copy
import numpy as np

import gridtools as gt
from namelist import cp, datatype, g, p_ref, Rd
from storages.grid_data import GridData

class DiagnosticIsentropic:
	"""
	Class implementing the diagnostic steps of the three-dimensional moist isentropic dynamical core
	using GT4Py's stencils.
	"""
	def __init__(self, grid, moist_on, backend):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.
		"""
		self._grid, self._moist_on, self._backend = grid, moist_on, backend

		# The pointers to the stencil's compute function.
		# They will be initialized the first time the entry-point methods are invoked.
		self._stencil_diagnosing_velocity_x = None
		self._stencil_diagnosing_velocity_y = None
		if self._moist_on:
			self._stencil_diagnosing_water_constituents_isentropic_density = None
			self._stencil_diagnosing_mass_fraction_of_water_constituents_in_air = None
		self._stencil_diagnosing_air_pressure = None
		self._stencil_diagnosing_montgomery = None
		self._stencil_diagnosing_height = None
		self._stencil_diagnosing_air_density = None
		self._stencil_diagnosing_air_temperature = None

		# Assign the corresponding z-level to each z-staggered grid point
		# This is required to diagnose the geometrical height at the half levels
		theta_1d = np.reshape(grid.z_half_levels.values[:, np.newaxis, np.newaxis], (1, 1, grid.nz + 1))
		self._theta = np.tile(theta_1d, (grid.nx, grid.ny, 1))
	
	@property
	def diagnostic(self):
		"""
		Get the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		if self._diagnostic is None:
			raise ValueError('''The attribute which is supposed to implement the diagnostic step of the moist isentroic ''' \
							 '''dynamical core is actually :obj:`None`. Please set it correctly.''')
		return self._diagnostic

	@diagnostic.setter
	def diagnostic(self, value):
		"""
		Set the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.

		Parameter
		---------
		value : obj
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		self._diagnostic = value

	def get_water_constituents_isentropic_density(self, state):
		"""
		Diagnosis of the isentropic density of each water constituent, i.e., :math:`Q_v`, :math:`Q_c` and :math:`Q_v`.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* air_isentropic_density (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* water_vapor_isentropic_density (unstaggered);
			* cloud_liquid_water_isentropic_density (unstaggered);
			* precipitation_water_isentropic_density (unstaggered).
		"""
		# Extract the required variables
		s  = state['air_isentropic_density'].values[:,:,:,0]
		qv = state['mass_fraction_of_water_vapor_in_air'].values[:,:,:,0]
		qc = state['mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0]
		qr = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_water_constituents_isentropic_density is None:
			self._initialize_stencil_diagnosing_water_constituents_isentropic_density()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_water_constituents_isentropic_density(s, qv, qc, qr)

		# Run the stencil's compute function
		self._stencil_diagnosing_water_constituents_isentropic_density.compute()

		# Set the output
		out = GridData(state.time, self._grid, 
					   water_vapor_isentropic_density = self._out_Qv,
					   cloud_liquid_water_isentropic_density = self._out_Qc, 
					   precipitation_water_isentropic_density = self._out_Qr)

		return out

	def get_velocity_components(self, state, state_old = None):
		"""
		Diagnosis of the velocity components :math:`u` and :math:`v`.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered).

		state_old : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables,
			defined at the previous time level:

			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered).

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered).

		Note
		----
		The first and last rows (respectively, columns) of the staggered :math:`x`-velocity (resp., :math:`y`-velocity) 
		are set only if the state at the previous time level is provided.
		"""
		# Extract the required variables at the current time level
		s = state['air_isentropic_density'].values[:,:,:,0]
		U = state['x_momentum_isentropic'].values[:,:,:,0]
		V = state['y_momentum_isentropic'].values[:,:,:,0]

		# Extract the required variables at the previous time level
		u_old = None if state_old['x_velocity'] is None else state_old['x_velocity'].values[:,:,:,0]
		v_old = None if state_old['y_velocity'] is None else state_old['y_velocity'].values[:,:,:,0]

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_velocity_x is None:
			self._initialize_stencil_diagnosing_velocity_x()
			self._initialize_stencil_diagnosing_velocity_y()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencils_diagnosing_velocity(s, U, V)

		# Run the stencils' compute functions
		self._stencil_diagnosing_velocity_x.compute()
		self._stencil_diagnosing_velocity_y.compute()

		# Possibly set the outermost layers
		if u_old is not None:	
			self.boundary.set_outermost_layers_x(self._out_u, u_old) 
		if v_old is not None:	
			self.boundary.set_outermost_layers_y(self._out_v, v_old) 

		# Set the output
		out = GridData(state.time, self._grid, 
					   x_velocity = self._out_u, 
					   y_velocity = self._out_v)

		return out

	def get_mass_fraction_of_water_constituents_in_air(self, state):
		"""
		Diagnosis of the mass fraction of each water constituents, i.e., :math:`q_v`, :math:`q_c` and :math:`q_r`.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* air_isentropic_density (unstaggered);
			* water_vapor_isentropic_density (unstaggered);
			* cloud_liquid_water_isentropic_density (unstaggered);
			* precipitation_water_isentropic_density (unstaggered).

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).
		"""	
		# Extract the required variables
		s  = state['air_isentropic_density'].values[:,:,:,0]
		Qv = state['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc = state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr = state['precipitation_water_isentropic_density'].values[:,:,:,0]

		# The first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil_diagnosing_mass_fraction_of_water_constituents_in_air is None:
			self._initialize_stencil_diagnosing_mass_fraction_of_water_constituents_in_air()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_mass_fraction_of_water_constituents_in_air(s, Qv, Qc, Qr)

		# Run the stencils' compute functions
		self._stencil_diagnosing_mass_fraction_of_water_constituents_in_air.compute()

		# Set the output
		out = GridData(state.time, self._grid, 
					   mass_fraction_of_water_vapor_in_air = self._out_qv,
					   mass_fraction_of_cloud_liquid_water_in_air = self._out_qc, 
					   mass_fraction_of_precipitation_water_in_air = self._out_qr)

		return out

	def get_diagnostic_variables(self, state, pt):
		"""
		Diagnosis of the pressure, the Exner function, the Montgomery potential, and the geometric height of the half-levels.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* air_isentropic_density (unstaggered).

		pt : float
			Pressure value at the top of the domain.

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height (:math:`z`-staggered).
		"""
		# Extract the required variables
		s  = state['air_isentropic_density'].values[:,:,:,0]

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_air_pressure is None:
			self._initialize_stencil_diagnosing_air_pressure()
			self._initialize_stencil_diagnosing_montgomery()
			self._initialize_stencil_diagnosing_height()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_air_pressure(s)

		# Apply upper boundary condition on pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_air_pressure.compute()
	
		# Compute the Exner function (not via a GT4Py's stencils)
		self._out_exn[:, :, :] = cp * (self._out_p[:, :, :] / p_ref) ** (Rd / cp) 

		# Compute Montgomery potential at the lower main level
		mtg_s = self._grid.z_half_levels.values[-1] * self._out_exn[:, :, -1] + g * self._grid.topography_height
		self._out_mtg[:, :, -1] = mtg_s + 0.5 * self._grid.dz * self._out_exn[:, :, -1]

		# Compute Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery.compute()

		# Compute geometrical height of isentropes
		self._out_h[:, :, -1] = self._grid.topography_height
		self._stencil_diagnosing_height.compute()

		# Set the output
		out = GridData(state.time, self._grid, 
					   air_pressure = self._out_p, 
					   exner_function = self._out_exn,
					   montgomery_potential = self._out_mtg, 
					   height = self._out_h)

		return out

	def get_air_density(self, state):
		"""
		Diagnosis of the density.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* air_isentropic_density (unstaggered);
			* height (:math:`z`-staggered).

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* air_density (unstaggered).
		"""
		# Extract the required variables
		s = state['air_isentropic_density'].values[:,:,:,0]
		h = state['height'].values[:,:,:,0]

		# If it is the first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil_diagnosing_air_density is None:
			self._initialize_stencil_diagnosing_air_density()

		# Update the attributes which serve as inputs to the stencil
		self._set_inputs_to_stencil_diagnosing_air_density(s, h)

		# Run the stencil's compute function
		self._stencil_diagnosing_air_density.compute()

		# Set the output
		out = GridData(state.time, self._grid, air_density = self._out_rho)

		return out

	def get_air_temperature(self, state):
		"""
		Diagnosis of the temperature.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes containing the following variables:

			* exner_function (:math:`z`-staggered).

		Return
		------
		obj :
			:class:`~storages.grid_data.GridData` collecting the diagnosed variables, namely:

			* air_temperature (unstaggered).
		"""
		# Extract the Exner function
		exn = state['exner_function'].values[:, :, :, -1]

		# Diagnose the temperature at the mass grid points (not via a GT4Py's stencil)
		T = .5 * (self._theta[:, :, :-1] * exn[:, :, :-1] + self._theta[:, :, 1:] * exn[:, :, 1:]) / cp

		# Set the output
		out = GridData(state.time, self._grid, air_temperature = T)

		return out


	def _initialize_stencil_diagnosing_water_constituents_isentropic_density(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the isentropic density of each water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_Qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_Qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_Qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_water_constituents_isentropic_density = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_water_constituents_isentropic_density,
			inputs = {'in_s': self._in_s, 'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr},
			outputs = {'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_water_constituents_isentropic_density(self, s, qv, qc, qr):	
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses 
		the isentropic density of each water constituent.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		qv : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of 
			water vapor.
		qc : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of
			cloud water.
		qr : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of
			precipitation water.
		"""
		self._in_s[:,:,:]  = s[:,:,:]
		self._in_qv[:,:,:] = qv[:,:,:]
		self._in_qc[:,:,:] = qc[:,:,:]
		self._in_qr[:,:,:] = qr[:,:,:]

	def _defs_stencil_diagnosing_water_constituents_isentropic_density(self, in_s, in_qv, in_qc, in_qr):
		"""
		GT4Py's stencil diagnosing the isentropic density of each water constituent, i.e., 
		:math:`Q_v`, :math:`Q_c` and :math:`Q_r`.

		Parameters
		----------
		in_s : obj 
			:class:`gridtools.Equation` representing the isentropic density.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapor.
		in_qc : obj 
			:class:`gridtools.Equation` representing the mass fraction of cloud water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		out_Qv : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qv`.
		out_Qc : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qc`.
		out_Qr : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qr`.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_Qv = gt.Equation()
		out_Qc = gt.Equation()
		out_Qr = gt.Equation()

		# Computations
		out_Qv[i, j, k] = in_s[i, j, k] * in_qv[i, j, k]
		out_Qc[i, j, k] = in_s[i, j, k] * in_qc[i, j, k]
		out_Qr[i, j, k] = in_s[i, j, k] * in_qr[i, j, k]

		return out_Qv, out_Qc, out_Qr


	def _initialize_stencil_diagnosing_velocity_x(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the :math:`x`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_U = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_u = np.zeros((nx + 1, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_x = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_x,
			inputs = {'in_s': self._in_s, 'in_U': self._in_U},
			outputs = {'out_u': self._out_u},
			domain = gt.domain.Rectangle((1, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _initialize_stencil_diagnosing_velocity_y(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the :math:`y`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_V = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_v = np.zeros((nx, ny + 1, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_y = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_y,
			inputs = {'in_s': self._in_s, 'in_V': self._in_V},
			outputs = {'out_v': self._out_v},
			domain = gt.domain.Rectangle((0, 1, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencils_diagnosing_velocity(self, s, U, V):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencils which diagnose 
		the velocity components.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		U : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
		V : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`y`-velocity.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_U[:,:,:] = U[:,:,:]
		self._in_V[:,:,:] = V[:,:,:]

	def _defs_stencil_diagnosing_velocity_x(self, in_s, in_U):
		"""
		GT4Py's stencil diagnosing the :math:`x`-component of the velocity.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`x`-velocity.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_u = gt.Equation()

		# Computations
		out_u[i, j, k] = (in_U[i-1, j, k] + in_U[i, j, k]) / (in_s[i-1, j, k] + in_s[i, j, k])

		return out_u

	def _defs_stencil_diagnosing_velocity_y(self, in_s, in_V):
		"""
		GT4Py's stencil diagnosing the :math:`y`-component of the velocity.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`y`-velocity.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_v = gt.Equation()

		# Computations
		out_v[i, j, k] = (in_V[i, j-1, k] + in_V[i, j, k]) / (in_s[i, j-1, k] + in_s[i, j, k])
			
		return out_v


	def _initialize_stencil_diagnosing_mass_fraction_of_water_constituents_in_air(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the mass fraction of each water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_mass_fraction_of_water_constituents_in_air = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_mass_fraction_of_water_constituents_in_air,
			inputs = {'in_s': self._in_s, 'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr},
			outputs = {'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_mass_fraction_of_water_constituents_in_air(self, s, Qv, Qc, Qr):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnose 
		the mass fraction of each water constituent.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		Qv : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapor.
		Qc : obj 
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		self._in_s[:,:,:]  = s[:,:,:]
		self._in_Qv[:,:,:] = Qv[:,:,:]
		self._in_Qc[:,:,:] = Qc[:,:,:]
		self._in_Qr[:,:,:] = Qr[:,:,:]


	def _defs_stencil_diagnosing_mass_fraction_of_water_constituents_in_air(self, in_s, in_Qv, in_Qc, in_Qr):
		"""
		GT4Py's stencil diagnosing the mass fraction of each water constituent.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		in_Qv : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapor.
		in_Qc : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		in_Qr : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.

		Returns
		-------
		out_qv : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of water vapor.
		out_qc : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of cloud water.
		out_qr : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()
		out_qr = gt.Equation()

		# Computations
		out_qv[i, j, k] = in_Qv[i, j, k] / in_s[i, j, k]
		out_qc[i, j, k] = in_Qc[i, j, k] / in_s[i, j, k]
		out_qr[i, j, k] = in_Qr[i, j, k] / in_s[i, j, k]

		return out_qv, out_qc, out_qr


	def _initialize_stencil_diagnosing_air_pressure(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the pressure.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the input field
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_p = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._in_p = self._out_p

		# Instantiate the stencil
		self._stencil_diagnosing_air_pressure = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_air_pressure,
			inputs = {'in_s': self._in_s, 'in_p': self._in_p},
			outputs = {'out_p': self._out_p},
			domain = gt.domain.Rectangle((0, 0, 1), (nx - 1, ny - 1, nz)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.FORWARD)

	def _initialize_stencil_diagnosing_montgomery(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the Montgomery potential.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the input and output field
		self._out_exn = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._out_mtg = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_mtg = self._out_mtg

		# Instantiate the stencil
		self._stencil_diagnosing_montgomery = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_montgomery,
			inputs = {'in_exn': self._out_exn, 'in_mtg': self._in_mtg},
			outputs = {'out_mtg': self._out_mtg},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 2)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.BACKWARD)
	
	def _initialize_stencil_diagnosing_height(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the geometric height of the half-level isentropes.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the output field
		self._out_h = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._in_h = self._out_h

		# Instantiate the stencil
		self._stencil_diagnosing_height = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_height,
			inputs = {'in_theta': self._theta, 'in_exn': self._out_exn, 'in_p': self._out_p, 'in_h': self._in_h},
			outputs = {'out_h': self._out_h},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.BACKWARD)

	def _set_inputs_to_stencil_diagnosing_air_pressure(self, s):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses the pressure.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		"""
		self._in_s[:,:,:] = s[:,:,:]
	
	def _defs_stencil_diagnosing_air_pressure(self, in_s, in_p):
		"""
		GT4Py's stencil diagnosing the pressure.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed pressure.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_p = gt.Equation()

		# Computations
		out_p[i, j, k] = in_p[i, j, k-1] + g * self._grid.dz * in_s[i, j, k-1]

		return out_p

	def _defs_stencil_diagnosing_montgomery(self, in_exn, in_mtg):
		"""
		GT4Py's stencil diagnosing the Exner function.

		Parameters
		----------
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed Montgomery potential.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_mtg = gt.Equation()

		# Computations
		out_mtg[i, j, k] = in_mtg[i, j, k+1] + self._grid.dz * in_exn[i, j, k+1]

		return out_mtg

	def _defs_stencil_diagnosing_height(self, in_theta, in_exn, in_p, in_h):
		"""
		GT4Py's stencil diagnosing the geometric height of the isentropes.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the vertical half levels.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the isentropes.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed geometric height of the isentropes.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_h = gt.Equation()

		# Computations
		out_h[i, j, k] = in_h[i, j, k+1] - Rd * (in_theta[i, j, k  ] * in_exn[i, j, k  ] +
												 in_theta[i, j, k+1] * in_exn[i, j, k+1]) * \
												(in_p[i, j, k] - in_p[i, j, k+1]) / \
												(cp * g * (in_p[i, j, k] + in_p[i, j, k+1]))

		return out_h


	def _initialize_stencil_diagnosing_air_density(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the density.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_h = np.zeros((nx, ny, nz + 1), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_rho = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_air_density = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_air_density,
			inputs = {'in_theta': self._theta, 'in_s': self._in_s, 'in_h': self._in_h},
			outputs = {'out_rho': self._out_rho},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_air_density(self, s, h):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses the density.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		h : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the height of the half-levels.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_h[:,:,:] = h[:,:,:]
	
	def _defs_stencil_diagnosing_air_density(self, in_theta, in_s, in_h):
		"""
		GT4Py's stencil diagnosing the density.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the vertical half levels.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height at the half-levels.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed density.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_rho = gt.Equation()

		# Computations
		out_rho[i, j, k] = in_s[i, j, k] * (in_theta[i, j, k] - in_theta[i, j, k+1]) / (in_h[i, j, k] - in_h[i, j, k+1])

		return out_rho
