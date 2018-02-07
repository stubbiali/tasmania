import numpy as np

from dycore.diffusion import Diffusion
from dycore.dycore import DynamicalCore
from dycore.horizontal_boundary import HorizontalBoundary, RelaxedSymmetricXZ, RelaxedSymmetricYZ
from dycore.diagnostic_isentropic import DiagnosticIsentropic
from dycore.prognostic_isentropic import PrognosticIsentropic
from dycore.vertical_damping import VerticalDamping
import gridtools as gt
from namelist import cp, datatype, g, p_ref, Rd
from storages.state_isentropic import StateIsentropic

class DynamicalCoreIsentropic(DynamicalCore):
	"""
	This class inherits :class:`~dycore.dycore.DynamicalCore` to implement the three-dimensional 
	(moist) isentropic dynamical core using GT4Py's stencils. The class offers different numerical
	schemes to carry out the prognostic step of the dynamical core, and supports different types of 
	lateral boundary conditions.
	"""
	def __init__(self, grid, imoist, horizontal_boundary_type, scheme, backend,
				 idamp = True, damp_type = 'rayleigh', damp_depth = 15, damp_max = 0.0002, 
				 idiff = True, diff_coeff = .03, diff_coeff_moist = .03, diff_max = .24):
		"""
		Constructor.

		Parameters
		----------
			grid : obj
				:class:`~grids.grid_xyz:GridXYZ` representing the underlying grid.
			imoist : bool
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			horizontal_boundary_type : str
				String specifying the horizontal boundary conditions. See :class:`~dycore.horizontal_boundary.HorizontalBoundary`
				for the available options.
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
		self._grid, self._imoist, self._idamp, self._idiff = grid, imoist, idamp, idiff

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = IsentropicPrognostic.factory(grid, imoist, scheme, backend)
		nb, self._tls = self._prognostic.nb, self._prognostic.time_levels

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = IsentropicDiagnostic(grid, imoist, backend)

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid, nb)

		# Instantiate the class in charge of applying vertical damping
		if idamp: 
			self._damper = VerticalDamping.factory(damp_type, grid, damp_depth, damp_max, backend)

		# Instantiate the classes in charge of applying numerical diffusion
		if idiff:
			self._diffuser = Diffusion(grid, idamp, damp_depth, diff_coeff, diff_max, backend)
			if imoist:
				self._diffuser_moist = Diffusion(grid, idamp, damp_depth, diff_coeff_moist, diff_max, backend)

		# Set pointer to the entry-point method, distinguishing between dry and moist model
		self._integrate = self._integrate_moist if imoist else self._integrate_dry

	def __call__(self, dt, state):
		"""
		Call operator advancing the state variables one step forward. 

		Parameters
		----------
			dt : obj 
				:class:`datetime.timedelta` representing the time step.
			state :obj 
				:class:`~storages.state_isentropic.StateIsentropic` representing the current state.

		Return
		------
			obj :
				:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		return self._integrate(dt, state)

	def get_initial_state(self, initial_time, x_velocity_initial, y_velocity_initial, brunt_vaisala_initial):
		"""
		Get the initial state, where:
			* :math:`u(x, \, y, \, 0) = u_0` and :math:`v(x, \, y, \, 0) = v_0`;
			* all the other model variables (Exner function, pressure, Montgomery potential, 
				height of the isentropes, isentropic density) are derived from the Brunt-Vaisala
				frequency :math:`N`.

		Parameters
		----------
			initial_time : obj 
				:class:`datetime.datetime` representing the initial simulation time.
			x_velocity_initial : float 
				The initial, uniform :math:`x`-velocity :math:`u_0`.
			y_velocity_initial : float 
				The initial, uniform :math:`y`-velocity :math:`v_0`.
			brunt_vaisala_initial : float
				The uniform Brunt-Vaisala frequence :math:`N`.

		Return
		------
			obj :
				:class:`~storages.state_isentropic.StateIsentropic` representing the initial state.
		"""
		if self._imoist:
			raise NotImplementedError()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		z, z_hl, dz = self._grid.z.values, self._grid.z_half_levels.values, self._grid.dz
		topo = self._grid.topography_height

		# The initial velocity
		u = x_velocity_initial * np.ones((nx + 1, ny, nz), dtype = datatype)
		v = y_velocity_initial * np.ones((nx, ny + 1, nz), dtype = datatype)

		# The initial Exner function
		exn_col = np.zeros(nz + 1, dtype = datatype)
		exn_col[-1] = cp
		for k in range(0, nz):
			exn_col[nz - k - 1] = exn_col[nz - k] - dz * g**2 / \
								  (brunt_vaisala_initial**2 * z[nz - k - 1]**2)
		exn = np.tile(exn_col[np.newaxis, np.newaxis, :], (nx, ny, 1))

		# The initial pressure
		p = p_ref * (exn / cp) ** (cp / Rd)

		# The initial Montgomery potential
		mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
		mtg = np.zeros((nx, ny, nz), dtype = datatype)
		mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
		for k in range(1, nz):
			mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

		# The initial geometrical height of the isentropes
		h = np.zeros((nx, ny, nz + 1), dtype = datatype)
		h[:, :, -1] = self._grid.topography_height
		for k in range(0, nz):
			h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * 2. * g / (brunt_vaisala_initial**2 * z[nz - k - 1])

		# The initial isentropic density
		s = - 1. / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

		# Assemble the initial state
		state = IsentropicState(initial_time, self._grid, s, u, v, p, exn, mtg, h)

		return state

	def _integrate_dry(self, dt, state):
		"""
		Entry-point method advancing the dry isentropic state by a single time step.

		Parameters
		----------
			dt : obj 
				:class:`datetime.timedelta` representing the time step.
			state :obj 
				:class:`~storages.state_isentropic.StateIsentropic` representing the current state.

		Return
		------
			obj :
				:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		# Extract the Numpy arrays carrying the current solution
		s_now   = state['isentropic_density'].values[:,:,:,0]
		u_now   = state['x_velocity'].values[:,:,:,0]
		v_now   = state['y_velocity'].values[:,:,:,0]
		p_now   = state['pressure'].values[:,:,:,0]
		exn_now = state['exner_function'].values[:,:,:,0]
		mtg_now = state['montgomery_potential'].values[:,:,:,0]
		h_now   = state['height'].values[:,:,:,0]

		# Diagnose the conservative model variables
		U_now, V_now = self._diagnostic.get_conservative_variables(s_now, u_now, v_now)

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_now_   = self._boundary.from_physical_to_computational_domain(s_now)
		u_now_   = self._boundary.from_physical_to_computational_domain(u_now)
		v_now_   = self._boundary.from_physical_to_computational_domain(v_now)
		mtg_now_ = self._boundary.from_physical_to_computational_domain(mtg_now)
		U_now_   = self._boundary.from_physical_to_computational_domain(U_now)
		V_now_   = self._boundary.from_physical_to_computational_domain(V_now)

		# If the time integrator is a two time-levels method and this is the first time step:
		# assume the old solution coincides with the current one
		if not hasattr(self, '_s_old_'):
			self._s_old_ = s_now_ if self._tls == 2 else None
			self._U_old_ = U_now_ if self._tls == 2 else None
			self._V_old_ = V_now_ if self._tls == 2 else None

		# Perform the prognostic step
		s_new_, U_new_, V_new_ = self._prognostic.step_forward(self._diagnostic, self._boundary, dt, 
															   s_now_, u_now_, v_now_, p_now, mtg_now_, U_now_, V_now_, 
															   old_s = self._s_old_, old_U = self._U_old_, old_V = self._V_old_)

		# Bring the vectors back to the original dimensions and/or apply the lateral boundary conditions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		s_new = self._boundary.from_computational_to_physical_domain(s_new_, (nx, ny, nz))
		if type(self._boundary) == RelaxedSymmetricXZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = False)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = True) 
		elif type(self._boundary) == RelaxedSymmetricYZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = True)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz))
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz)) 
		
		self._boundary.apply(s_new, s_now)
		self._boundary.apply(U_new, U_now)
		self._boundary.apply(V_new, V_now)

		# Apply vertical damping
		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref = s_now
				self._U_ref = U_now
				self._V_ref = V_now

			s_new[:,:,:] = self._damper.apply(self._tls * dt, s_now, s_new, self._s_ref)
			U_new[:,:,:] = self._damper.apply(self._tls * dt, U_now, U_new, self._U_ref)
			V_new[:,:,:] = self._damper.apply(self._tls * dt, V_now, V_new, self._V_ref)

		# Apply numerical diffusion
		if self._idiff:
			s_new[:,:,:] = self._diffuser.apply(s_new)
			U_new[:,:,:] = self._diffuser.apply(U_new)
			V_new[:,:,:] = self._diffuser.apply(V_new)

		# Diagnose the non-conservative model variables
		u_new, v_new = self._diagnostic.get_nonconservative_variables(s_new, U_new, V_new)

		# Apply the lateral boundary conditions to the velocity components
		self._boundary.set_outermost_layers_x(u_new, u_now) 
		self._boundary.set_outermost_layers_y(v_new, v_now) 

		# Diagnose the pressure, the Exner function, the Montgomery potential and 
		# the geometrical height at the half levels
		p_new, exn_new, mtg_new, h_new = self._diagnostic.get_diagnostic_variables(s_new, p_now[0,0,0])

		# Update the old time step
		if self._tls == 2:
			self._s_old_[:,:,:] = s_now_
			self._U_old_[:,:,:] = U_now_
			self._V_old_[:,:,:] = V_now_

		# Build up the new state, and return
		state_new = IsentropicState(state.time + dt, self._grid,
									s_new, u_new, v_new, p_new, exn_new, mtg_new, h_new)

		return state_new

	def _integrate_moist(self, dt, state):
		"""
		Entry-point method advancing the moist isentropic state by a single time step.

		Parameters
		----------
			dt : obj 
				:class:`datetime.timedelta` representing the time step.
			state :obj 
				:class:`~storages.state_isentropic.StateIsentropic` representing the current state.

		Return
		------
			obj :
				:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		# Extract the Numpy arrays carrying the current solution
		s_now   = state['isentropic_density'].values[:,:,:,0]
		u_now   = state['x_velocity'].values[:,:,:,0]
		v_now   = state['y_velocity'].values[:,:,:,0]
		p_now   = state['pressure'].values[:,:,:,0]
		exn_now = state['exner_function'].values[:,:,:,0]
		mtg_now = state['montgomery_potential'].values[:,:,:,0]
		h_now   = state['height'].values[:,:,:,0]
		qv_now  = state['water_vapour'].values[:,:,:,0]
		qc_now  = state['cloud_water'].values[:,:,:,0]
		qr_now  = state['precipitation_water'].values[:,:,:,0]

		# Diagnose the conservative model variables
		U_now, V_now, Qv_now, Qc_now, Qr_now = \
			self._diagnostic.get_conservative_variables(s_now, u_now, v_now, qv_now, qc_now, qr_now)

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_now_   = self._boundary.from_physical_to_computational_domain(s_now)
		u_now_   = self._boundary.from_physical_to_computational_domain(u_now)
		v_now_   = self._boundary.from_physical_to_computational_domain(v_now)
		mtg_now_ = self._boundary.from_physical_to_computational_domain(mtg_now)
		U_now_   = self._boundary.from_physical_to_computational_domain(U_now)
		V_now_   = self._boundary.from_physical_to_computational_domain(V_now)
		Qv_now_  = self._boundary.from_physical_to_computational_domain(Qv_now)
		Qc_now_  = self._boundary.from_physical_to_computational_domain(Qc_now)
		Qr_now_  = self._boundary.from_physical_to_computational_domain(Qr_now)

		# If the time integrator is a two time-levels method and this is the first time step:
		# assume the old solution coincides with the current one
		if not hasattr(self, '_s_old_'):
			self._s_old_  = s_now_  if self._tls == 2 else None
			self._U_old_  = U_now_  if self._tls == 2 else None
			self._V_old_  = V_now_  if self._tls == 2 else None
			self._Qv_old_ = Qv_now_ if self._tls == 2 else None
			self._Qc_old_ = Qc_now_ if self._tls == 2 else None
			self._Qr_old_ = Qr_now_ if self._tls == 2 else None

		# Perform the prognostic step
		s_new, U_new, V_new, Qv_new, Qc_new, Qr_new = \
			self._prognostic.step_forward(self._diagnostic, self._boundary, dt, s_now_, u_now_, v_now_, 
										  p_now, mtg_now_, U_now_, V_now_, Qv_now_, Qc_now_, Qr_now_,
										  self._s_old_, self._U_old_, self._V_old_,
										  self._Qv_old_, self._Qc_old_, self._Qr_old_)

		# Apply vertical damping
		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref_'):
				self._s_ref_  = s_now_
				self._U_ref_  = U_now_
				self._V_ref_  = V_now_
				self._Qv_ref_ = Qv_now_
				self._Qc_ref_ = Qc_now_
				self._Qr_ref_ = Qr_now_

			s_new_[:,:,:]  = self._damper.apply(self._tls * dt, s_now_ , s_new_ , self._s_ref_ )
			U_new_[:,:,:]  = self._damper.apply(self._tls * dt, U_now_ , U_new_ , self._U_ref_ )
			V_new_[:,:,:]  = self._damper.apply(self._tls * dt, V_now_ , V_new_ , self._V_ref_ )
			Qv_new_[:,:,:] = self._damper.apply(self._tls * dt, Qv_now_, Qv_new_, self._Qv_ref_)
			Qc_new_[:,:,:] = self._damper.apply(self._tls * dt, Qc_now_, Qc_new_, self._Qc_ref_)
			Qr_new_[:,:,:] = self._damper.apply(self._tls * dt, Qr_now_, Qr_new_, self._Qr_ref_)

		# Apply numerical diffusion
		if self._idiff:
			s_new_[:,:,:]  = self._diffuser.apply(s_new_)
			U_new_[:,:,:]  = self._diffuser.apply(U_new_)
			V_new_[:,:,:]  = self._diffuser.apply(V_new_)
			Qv_new_[:,:,:] = self._diffuser_moist.apply(Qv_new_)
			Qc_new_[:,:,:] = self._diffuser_moist.apply(Qc_new_)
			Qr_new_[:,:,:] = self._diffuser_moist.apply(Qr_new_)

		# Bring the vectors back to the original dimensions and/or apply the lateral boundary conditions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		s_new = self._boundary.from_computational_to_physical_domain(s_new_, (nx, ny, nz))
		if type(self._boundary) == RelaxedSymmetricXZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = False)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = True) 
		elif type(self._boundary) == RelaxedSymmetricYZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = True)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz))
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz)) 
		Qv_new = self._boundary.from_computational_to_physical_domain(Qv_new_, (nx, ny, nz)) 
		Qc_new = self._boundary.from_computational_to_physical_domain(Qc_new_, (nx, ny, nz)) 
		Qr_new = self._boundary.from_computational_to_physical_domain(Qr_new_, (nx, ny, nz)) 
		
		self._boundary.apply(s_new , s_now )
		self._boundary.apply(U_new , U_now )
		self._boundary.apply(V_new , V_now )
		self._boundary.apply(Qv_new, Qv_now)
		self._boundary.apply(Qc_new, Qc_now)
		self._boundary.apply(Qr_new, Qr_now)

		# Diagnose the non-conservative model variables
		u_new, v_new, qv_new, qc_new, qr_new = \
			self._diagnostic.get_nonconservative_variables(s_new, U_new, V_new, Qv_new, Qc_new, Qr_new)

		# Apply the lateral boundary conditions to the velocity components
		self._boundary.set_outermost_layers_x(u_new, u_now) 
		self._boundary.set_outermost_layers_y(v_new, v_now) 

		# Diagnose the pressure, the Exner function, the Montgomery potential, and 
		# the geometrical height at the half levels
		p_new, exn_new, mtg_new, h_new = self._diagnostic.get_diagnostic_variables(s_new, p_now[0,0,0])

		# Update the old time step
		if self._tls == 2:
			self._s_old_  = s_now_
			self._U_old_  = U_now_
			self._V_old_  = V_now_
			self._Qv_old_ = Qv_now_
			self._Qc_old_ = Qc_now_
			self._Qr_old_ = Qr_now_

		# Build up the new state, and return
		state_new = IsentropicState(state.time + dt, self._grid,
									s_new, u_new, v_new, p_new, exn_new, mtg_new, h_new, qv_new, qc_new, qr_new)

		return state_new
		
