import numpy as np

from dycore.diagnostic_isentropic import DiagnosticIsentropic
from namelist import cp, g, p_ref, Rd

class DiagnosticIsentropicIsothermal(DiagnosticIsentropic):
	"""
	This class inherits :class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` to implement the 
	diagnostic steps of the three-dimensional moist isentropic and isothernal dynamical core using GT4Py's stencils.
	"""
	def __init__(self, grid, moist_on, backend):
		"""
		Constructor.

		Parameters
		----------
		temperature : float
			The temperature ([:math:`K`]).
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
		self._stencil_diagnosing_conservative_variables = None
		self._stencil_diagnosing_velocity_x = None
		self._stencil_diagnosing_velocity_y = None
		if self._moist_on:
			self._stencil_diagnosing_water_constituents = None

		# Assign the corresponding z-level to each grid point
		self._theta_s  = grid.z_half_levels[-1]
		theta_1d       = np.reshape(self._grid.z.values[:, np.newaxis, np.newaxis], (1, 1, grid.nz))
		theta_1d_hl    = np.reshape(self._grid.z_half_levels.values[:, np.newaxis, np.newaxis], (1, 1, grid.nz + 1))
		self._theta    = np.tile(theta_1d, (grid.nx, grid.ny, 1))
		self._theta_hl = np.tile(theta_1d_hl, (grid.nx, grid.ny, 1))

		# Initialize the attribute representing the temperature value
		self._temperature = None

	@property
	def temperature(self):
		return self._temperature

	@temperature.setter
	def temperature(self, T):
		self._temperature = T

	def get_diagnostic_variables(self, s, pt = None):
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
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			pressure.
		exn : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			Exner function.
		mtg : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed 
			Montgomery potential.
		h : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			geometric height of the potential temperature surfaces.
		"""
		# Compute the Pressure distribution
		self._out_p   = g * Rd * s * self._theta / cp

		# Compute the Exner function
		self._out_exn = cp * (self._out_p / p_ref) ** (Rd / cp)

		# Compute the Montgomery potential
		T = self._temperature
		self._out_mtg = cp * T + \
						g * np.repeat(self._grid.topography_height[:, :, np.newaxis], self._grid.nz, axis = 2) + \
						cp * T * np.log(self._theta / self._theta_s)

		# Compute the height of the isentropes
		self._out_h   = np.repeat(self._grid.topography_height[:, :, np.newaxis], self._grid.nz + 1, axis = 2) + \
					    cp * T / g * np.log(self._theta_hl / self._theta_s)

		return self._out_p, self._out_exn, self._out_mtg, self._out_h
