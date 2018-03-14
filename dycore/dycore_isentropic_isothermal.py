import numpy as np

from dycore.diffusion import Diffusion
from dycore.dycore_isentropic import DynamicalCoreIsentropic
from dycore.horizontal_boundary import HorizontalBoundary, RelaxedSymmetricXZ, RelaxedSymmetricYZ
from dycore.diagnostic_isentropic_isothermal import DiagnosticIsentropicIsothermal
from dycore.prognostic_isentropic import PrognosticIsentropic
from dycore.vertical_damping import VerticalDamping
from namelist import cp, datatype, g, p_ref, Rd
from storages.state_isentropic import StateIsentropic

class DynamicalCoreIsentropicIsothermal(DynamicalCoreIsentropic):
	"""
	This class inherits :class:`~dycore.dycore_isentropic.DynamicalCoreIsentropic` to implement the three-dimensional 
	(moist) isentropic and isothermal dynamical core using GT4Py's stencils. The class offers different numerical
	schemes to carry out the prognostic step of the dynamical core, and supports different types of lateral boundary conditions.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_type, scheme, backend,
				 idamp = True, damp_type = 'rayleigh', damp_depth = 15, damp_max = 0.0002, 
				 idiff = True, diff_coeff = .03, diff_coeff_moist = .03, diff_max = .24):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions. 
			See :class:`~dycore.horizontal_boundary.HorizontalBoundary` for the available options.
		scheme : str
			String specifying the numerical scheme carrying out the prognostic step of the dynamical core.
			See :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` for the available options.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils implementing the dynamical core.
		idamp : `bool`, optional
			:obj:`True` if vertical damping is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. Default is 'rayleigh'.
			See :class:`dycore.vertical_damping.VerticalDamping` for further details.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Default is 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Default is 0.0002.
		idiff : `bool`, optional
			:obj:`True` if numerical diffusion is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		diff_coeff : `float`, optional
			Diffusion coefficient. Default is 0.03.
		diff_coeff_moist : `float`, optional
			Diffusion coefficient for the water constituents. Default is 0.03.
		diff_max : `float`, optional
			Maximum value for the diffusion coefficient. Default is 0.24. See :class:`~dycore.diffusion.Diffusion`
			for further details.
		"""
		# Set parameters as attributes
		self._grid, self._moist_on, self._idamp, self._idiff = grid, moist_on, idamp, idiff

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = PrognosticIsentropic.factory(grid, moist_on, scheme, backend)
		nb, self._tls = self._prognostic.nb, self._prognostic.time_levels

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = DiagnosticIsentropicIsothermal(grid, moist_on, backend)

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid, nb)

		# Instantiate the class in charge of applying vertical damping
		if idamp: 
			self._damper = VerticalDamping.factory(damp_type, grid, damp_depth, damp_max, backend)

		# Instantiate the classes in charge of applying numerical diffusion
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if idiff:
			self._diffuser = Diffusion.factory((nx, ny, nz), grid, idamp, damp_depth, diff_coeff, diff_max, backend)
			if moist_on:
				self._diffuser_moist = Diffusion.factory((nx, ny, nz), grid, idamp, damp_depth, 
														 diff_coeff_moist, diff_max, backend)

		# Set pointer to the entry-point method, distinguishing between dry and moist model
		self._integrate = self._integrate_moist if moist_on else self._integrate_dry

	def get_initial_state(self, initial_time, initial_state_type = 0, **kwargs):
		"""
		Get the initial state, so that:

		* :math:`u(x, \, y, \, \\theta, \, 0) = u_0` and :math:`v(x, \, y, \, \\theta, \, 0) = v_0`;
		* :math:`T(x, \, y, \, \\theta, \, 0) = T_0`.

		Parameters
		----------
		initial_time : obj 
			:class:`datetime.datetime` representing the initial simulation time.
		initial_state_type : int
			The initial state identifier. By the time being, this is a dummy parameter, considered
			only to ensure compliancy with the class hierarchy interface.

		Keyword arguments
		-----------------
		x_velocity_initial : float 
			The initial, uniform :math:`x`-velocity :math:`u_0`. Default is :math:`10 m s^{-1}`.
		y_velocity_initial : float 
			The initial, uniform :math:`y`-velocity :math:`v_0`. Default is :math:`0 m s^{-1}`.
		temperature : float
			The uniform temperature :math:`T_0`. Default is :math:`250 K`.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` representing the initial state.
		"""
		if self._moist_on:
			raise NotImplementedError()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		z, z_hl, dz = self._grid.z.values, self._grid.z_half_levels.values, self._grid.dz
		topo = self._grid.topography_height

		# Set the temperature within the diagnostic component
		self._diagnostic.temperature = kwargs.get('temperature', 250.)

		# The initial velocity
		u = kwargs.get('x_velocity_initial', 10.) * np.ones((nx + 1, ny, nz), dtype = datatype)
		v = kwargs.get('y_velocity_initial', 0.) * np.ones((nx, ny + 1, nz), dtype = datatype)

		# The initial pressure
		p = p_ref * (kwargs.get('temperature', 250.) / \
			np.tile(z_hl[np.newaxis, np.newaxis, :], (nx, ny, 1))) ** (cp / Rd)

		# The initial Exner function
		exn = cp * (p / p_ref) ** (Rd / cp)

		# The initial geometric height of the isentropes
		h    = np.repeat(topo[:, :, np.newaxis], nz + 1, axis = 2) + cp * kwargs.get('temperature', 250.) / g * \
			   np.log(np.tile(z_hl[np.newaxis, np.newaxis, :], (nx, ny, 1)) / z_hl[-1])
		h_ml = np.repeat(topo[:, :, np.newaxis], nz, axis = 2) + cp * kwargs.get('temperature', 250.) / g * \
			   np.log(np.tile(z[np.newaxis, np.newaxis, :], (nx, ny, 1)) / z_hl[-1])

		# The initial Montgomery potential
		mtg = cp * kwargs.get('temperature', 250.) + g * h_ml

		# The initial isentropic density
		s = - 1. / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

		# The initial momentums
		U = s * kwargs.get('x_velocity_initial', 10.)
		V = s * kwargs.get('y_velocity_initial', 0.)

		# Assemble the initial state
		state = StateIsentropic(initial_time, self._grid, s, u, U, v, V, p, exn, mtg, h)

		return state
