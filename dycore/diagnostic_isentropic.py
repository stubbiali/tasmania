import numpy as np

import gridtools as gt
from namelist import cp, datatype, g, p_ref, Rd

class DiagnosticIsentropic:
	"""
	Class implementing the diagnostic steps of the three-dimensional moist isentropic dynamical core
	using GT4Py's stencils.
	"""
	def __init__(self, grid, imoist, backend):
		"""
		Constructor.

		Parameters
		----------
			grid : obj
				:class:`~grids.xyz_grid.XYZGrid` representing the underlying grid.
			imoist : bool 
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			backend : obj 
				:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.
		"""
		self._grid, self._imoist, self._backend = grid, imoist, backend

		# The pointers to the stencil's compute function.
		# They will be initialized the first time the entry-point methods are invoked.
		self._stencil_diagnosing_conservative_variables = None
		self._stencil_diagnosing_velocity_x = None
		self._stencil_diagnosing_velocity_y = None
		if self._imoist:
			self._stencil_diagnosing_water_constituents = None
		self._stencil_diagnosing_pressure = None
		self._stencil_diagnosing_montgomery = None
		self._stencil_diagnosing_height = None

		# Allocate the Numpy array which will store the Exner function
		# Conversely to all other Numpy arrays carrying the output fields, this array is allocated
		# here as the Exner function, being a nonlinear function of the pressure,  can not be diagnosed 
		# via a GT4Py's stencil
		self._out_exn = np.zeros((grid.nx, grid.ny, grid.nz + 1), dtype = datatype)

		# Assign the corresponding z-level to each z-staggered grid point
		# This is required to diagnose the geometrical height at the half levels
		theta_1d = np.reshape(grid.z_half_levels.values[:, np.newaxis, np.newaxis], (1, 1, grid.nz + 1))
		self._theta = np.tile(theta_1d, (grid.nx, grid.ny, 1))

	def get_conservative_variables(self, s, u, v, qv = None, qc = None, qr = None):
		"""
		Diagnosis of the conservative model variables, i.e., the momentums - :math:`U` and :math:`V` -
		and, optionally, the mass of water constituents - :math:`Q_v`, :math:`Q_c` and :math:`Q_r`.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
			u : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
			v : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the :math:`y`-velocity.
			qv : `array_like`, optional 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					water vapour.
			qc : `array_like`, optional 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					cloud water.
			qr : `array_like`, optional 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					precipitation water.

		Returns
		-------
			U : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`U`.
			V : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`V`.
			Qv : `array_like`, optional 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_v`.
			Qc : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_c`.
			Qr : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_r`.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_conservative_variables is None:
			self._initialize_stencil_diagnosing_conservative_variables()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_conservative_variables(s, u, v, qv, qc, qr)

		# Run the stencil's compute function
		self._stencil_diagnosing_conservative_variables.compute()

		if self._imoist:
			return self._out_U, self._out_V, self._out_Qv, self._out_Qc, self._out_Qr
		return self._out_U, self._out_V

	def get_nonconservative_variables(self, s, U, V, Qv = None, Qc = None, Qr = None):
		"""
		Diagnosis of the non-conservative model variables, i.e., the velocity components - :math:`u` and :math:`v` - 
		and, optionally, the mass fraction of the water constituents - :math:`q_v`, :math:`q_c` and :math:`q_r`.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
			U : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
			V : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`y`-velocity.
			Qv : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of water vapour.
			Qc : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of cloud water.
			Qr : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of precipitation water.

		Returns
		-------
			u : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`u`.
			v : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the diagnosed :math:`v`.
			qv : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_v`.
			qc : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_c`.
			qr : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_r`.

		Note
		----
			The first and last rows (respectively, columns) of :data:`u` (resp., :data:`v`) are not set by the method.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_velocity_x is None:
			self._initialize_stencil_diagnosing_velocity_x()
			self._initialize_stencil_diagnosing_velocity_y()
			if self._imoist:
				self._initialize_stencil_diagnosing_water_constituents()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencils_diagnosing_nonconservative_variables(s, U, V, Qv, Qc, Qr)

		# Run the stencils' compute functions
		self._stencil_diagnosing_velocity_x.compute()
		self._stencil_diagnosing_velocity_y.compute()
		if self._imoist:
			self._stencil_diagnosing_water_constituents.compute()
			return self._out_u, self._out_v, self._out_qv, self._out_qc, self._out_qr
		return self._out_u, self._out_v

	def get_diagnostic_variables(self, s, pt):
		"""
		Diagnosis of the pressure, the Exner function, the Montgomery potential, and the geometric height of the 
		potential temperature surfaces.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
			pt : float 
				Boundary value for the pressure at the top of the domain.

		Returns
		-------
			p : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed \
					pressure.
			exn : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed \
					Exner function.
			mtg : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed \
					Montgomery potential.
			h : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed \
					geometric height of the potential temperature surfaces.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_pressure is None:
			self._initialize_stencil_diagnosing_pressure()
			self._initialize_stencil_diagnosing_montgomery()
			self._initialize_stencil_diagnosing_height()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_pressure(s)

		# Apply upper boundary condition for pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_pressure.compute()
	
		# Compute the Exner function
		# Note: the Exner function can not be computed via a GT4Py's stencils as it is a
		# nonlinear function of the pressure distribution
		self._out_exn[:, :, :] = cp * (self._out_p[:, :, :] / p_ref) ** (Rd / cp) 

		# Compute Montgomery potential at the lower main level
		mtg_s = self._grid.z_half_levels.values[-1] * self._out_exn[:, :, -1] + g * self._grid.topography_height
		self._out_mtg[:, :, -1] = mtg_s + 0.5 * self._grid.dz * self._out_exn[:, :, -1]

		# Compute Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery.compute()

		# Compute geometrical height of isentropes
		self._out_h[:, :, -1] = self._grid.topography_height
		self._stencil_diagnosing_height.compute()

		return self._out_p, self._out_exn, self._out_mtg, self._out_h


	def _initialize_stencil_diagnosing_conservative_variables(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the conservative model variables.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_u = np.zeros((nx + 1, ny, nz), dtype = datatype)
		self._in_v = np.zeros((nx, ny + 1, nz), dtype = datatype)
		if self._imoist:
			self._in_qv = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_qc = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_U = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_V = np.zeros((nx, ny, nz), dtype = datatype)
		if self._imoist:
			self._out_Qv = np.zeros((nx, ny, nz), dtype = datatype)
			self._out_Qc = np.zeros((nx, ny, nz), dtype = datatype)
			self._out_Qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Set the computational domain and the domain
		_domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1))
		_mode = self._backend

		# Instantiate the stencil
		if not self._imoist:
			self._stencil_diagnosing_conservative_variables = gt.NGStencil( 
				definitions_func = self._defs_stencil_diagnosing_conservative_variables,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v},
				outputs = {'out_U': self._out_U, 'out_V': self._out_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_diagnosing_conservative_variables = gt.NGStencil( 
				definitions_func = self._defs_stencil_diagnosing_conservative_variables,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr},
				outputs = {'out_U': self._out_U, 'out_V': self._out_V,
						   'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
				domain = _domain, 
				mode = _mode)

	def _set_inputs_to_stencil_diagnosing_conservative_variables(self, s, u, v, qv, qc, qr):	
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses 
		the conservative variables.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
			u : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
			v : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the :math:`y`-velocity.
			qv : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					water vapour.
			qc : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					cloud water.
			qr : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of \
					precipitation water.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_u[:,:,:] = u[:,:,:]
		self._in_v[:,:,:] = v[:,:,:]
		if self._imoist:
			self._in_qv[:,:,:] = qv[:,:,:]
			self._in_qc[:,:,:] = qc[:,:,:]
			self._in_qr[:,:,:] = qr[:,:,:]

	def _defs_stencil_diagnosing_conservative_variables(self, in_s, in_u, in_v, 
														in_qv = None, in_qc = None, in_qr = None):
		"""
		GT4Py's stencil diagnosing the conservative model variables, i.e., the momentums - :math:`U` and :math:`V` -
		and, optionally, the mass of water constituents - :math:`Q_v`, :math:`Q_c` and :math:`Q_r`.

		Parameters
		----------
			in_s : obj 
				:class:`gridtools.Equation` representing the isentropic density.
			in_u : obj
				:class:`gridtools.Equation` representing the :math:`x`-velocity.
			in_v : obj 
				:class:`gridtools.Equation` representing the :math:`y`-velocity.
			in_qv : `obj`, optional
				:class:`gridtools.Equation` representing the mass fraction of water vapour.
			in_qc : `obj`, optional 
				:class:`gridtools.Equation` representing the mass fraction of cloud water.
			in_qr : `obj`, optional
				:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
			out_U : obj
				:class:`gridtools.Equation` representing the diagnosed :math:`U`.
			out_V : obj
				:class:`gridtools.Equation` representing the diagnosed :math:`V`.
			out_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the diagnosed :math:`Qv`.
			out_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the diagnosed :math:`Qc`.
			out_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the diagnosed :math:`Qr`.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._imoist:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		out_U[i, j, k] = 0.5 * in_s[i, j, k] * (in_u[i, j, k] + in_u[i+1, j, k])
		out_V[i, j, k] = 0.5 * in_s[i, j, k] * (in_v[i, j, k] + in_v[i, j+1, k])
		if self._imoist:
			out_Qv[i, j, k] = in_s[i, j, k] * in_qv[i, j, k]
			out_Qc[i, j, k] = in_s[i, j, k] * in_qc[i, j, k]
			out_Qr[i, j, k] = in_s[i, j, k] * in_qr[i, j, k]

			return out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_U, out_V


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

		# Set the computational domain
		_domain = gt.domain.Rectangle((1, 0, 0), (nx - 1, ny - 1, nz - 1))

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_x = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_x,
			inputs = {'in_s': self._in_s, 'in_U': self._in_U},
			outputs = {'out_u': self._out_u},
			domain = _domain, 
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

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 1, 0), (nx - 1, ny - 1, nz - 1))

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_y = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_y,
			inputs = {'in_s': self._in_s, 'in_V': self._in_V},
			outputs = {'out_v': self._out_v},
			domain = _domain, 
			mode = self._backend)

	def _initialize_stencil_diagnosing_water_constituents(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the water constituents.
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

		# Set the computational time
		_domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1))

		# Instantiate the stencil
		self._stencil_diagnosing_water_constituents = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_water_constituents,
			inputs = {'in_s': self._in_s, 'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr},
			outputs = {'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr},
			domain = _domain, 
			mode = self._backend)

	def _set_inputs_to_stencils_diagnosing_nonconservative_variables(self, s, U, V, Qv, Qc, Qr):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencils which diagnose 
		the nonconservative variables.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
			U : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
			V : array_like 
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`y`-velocity.
			Qv : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of water vapour.
			Qc : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of cloud water.
			Qr : `array_like`, optional
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of precipitation water.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_U[:,:,:] = U[:,:,:]
		self._in_V[:,:,:] = V[:,:,:]
		if self._imoist:
			self._in_Qv[:,:,:] = Qv[:,:,:]	
			self._in_Qc[:,:,:] = Qc[:,:,:]	
			self._in_Qr[:,:,:] = Qr[:,:,:]	

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

	def _defs_stencil_diagnosing_water_constituents(self, in_s, in_U, in_V, in_Qv, in_Qc, in_Qr):
		"""
		GT4Py's stencil diagnosing the water constituents.

		Parameters
		----------
			in_s : obj
				:class:`gridtools.Equation` representing the isentropic density.
			in_U : obj
				:class:`gridtools.Equation` representing the :math:`x`-momentum.
			in_V : obj
				:class:`gridtools.Equation` representing the :math:`y`-momentum.
			in_Qv : obj
				:class:`gridtools.Equation` representing the mass of water vapour.
			in_Qc : obj
				:class:`gridtools.Equation` representing the mass of cloud water.
			in_Qr : obj
				:class:`gridtools.Equation` representing the mass of precipitation water.

		Returns
		-------
			out_qv : obj
				:class:`gridtools.Equation` representing the diagnosed mass fraction of water vapour.
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


	def _initialize_stencil_diagnosing_pressure(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the pressure.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the output field
		self._out_p = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._in_p = self._out_p

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 0, 1), (nx - 1, ny - 1, nz))

		# Instantiate the stencil
		self._stencil_diagnosing_pressure = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_pressure,
			inputs = {'in_s': self._in_s, 'in_p': self._in_p},
			outputs = {'out_p': self._out_p},
			domain = _domain,
			mode = self._backend,
			vertical_direction = gt.vertical_direction.FORWARD)

	def _initialize_stencil_diagnosing_montgomery(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the Montgomery potential.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the output field
		self._out_mtg = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_mtg = self._out_mtg

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 2))

		self._stencil_diagnosing_montgomery = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_montgomery,
			inputs = {'in_exn': self._out_exn, 'in_mtg': self._in_mtg},
			outputs = {'out_mtg': self._out_mtg},
			domain = _domain,
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

		# Set computational domain
		_domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1))

		# Instantiate the stencil
		self._stencil_diagnosing_height = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_height,
			inputs = {'in_theta': self._theta, 'in_exn': self._out_exn, 'in_p': self._out_p, 'in_h': self._in_h},
			outputs = {'out_h': self._out_h},
			domain = _domain,
			mode = self._backend,
			vertical_direction = gt.vertical_direction.BACKWARD)
	
	def _set_inputs_to_stencil_diagnosing_pressure(self, s):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses the pressure.

		Parameters
		----------
			s : array_like
				:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		"""
		self._in_s[:,:,:] = s[:,:,:]
	
	def _defs_stencil_diagnosing_pressure(self, in_s, in_p):
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

