import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic

class FluxIsentropicCentered(FluxIsentropic):
	"""
	Class which inherits :class:`~dycore.flux_isentropic.FluxIsentropic` to implement a centered scheme to compute 
	the numerical fluxes for the governing equations expressed in conservative form using isentropic coordinates.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1

	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the centered :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Compute fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_centered_flux_x(i, j, k, u_now, s_now)
		self._flux_s_y = self._get_centered_flux_y(i, j, k, v_now, s_now)
		self._flux_U_x = self._get_centered_flux_x(i, j, k, u_now, U_now)
		self._flux_U_y = self._get_centered_flux_y(i, j, k, v_now, U_now)
		self._flux_V_x = self._get_centered_flux_x(i, j, k, u_now, V_now)
		self._flux_V_y = self._get_centered_flux_y(i, j, k, v_now, V_now)
		
		if self._moist_on:
			# Compute fluxes for the water constituents
			self._flux_Qv_x = self._get_centered_flux_x(i, j, k, u_now, Qv_now)
			self._flux_Qv_y = self._get_centered_flux_y(i, j, k, v_now, Qv_now)
			self._flux_Qc_x = self._get_centered_flux_x(i, j, k, u_now, Qc_now)
			self._flux_Qc_y = self._get_centered_flux_y(i, j, k, v_now, Qc_now)
			self._flux_Qr_x = self._get_centered_flux_x(i, j, k, u_now, Qr_now)
			self._flux_Qr_y = self._get_centered_flux_y(i, j, k, v_now, Qr_now)

	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the centered :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k-1])

		# Compute flux for the isentropic density and the momentums
		self._flux_s_z = self._get_upwind_flux_z(i, j, k, w_mid, s_now)
		self._flux_U_z = self._get_upwind_flux_z(i, j, k, w_mid, U_now)
		self._flux_V_z = self._get_upwind_flux_z(i, j, k, w_mid, V_now)
		
		if self._moist_on:
			# Compute flux for the water constituents
			self._flux_Qv_z = self._get_upwind_flux_z(i, j, k, w_mid, Qv_now)
			self._flux_Qc_z = self._get_upwind_flux_z(i, j, k, w_mid, Qc_now)
			self._flux_Qr_z = self._get_upwind_flux_z(i, j, k, w_mid, Qr_now)

	def _get_centered_flux_x(self, i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`x`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`x`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = u[i+1, j, k] * 0.5 * (phi[i, j, k] + phi[i+1, j, k])
		return flux

	def _get_centered_flux_y(self, i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`y`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`y`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = v[i, j+1, k] * 0.5 * (phi[i, j, k] + phi[i, j+1, k])
		return flux
	
	def _get_centered_flux_z(self, i, j, k, w_mid, phi_now):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`z`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		w_mid : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in
			potential temperature, at the model half levels.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`z`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = w_mid[i, j, k] * 0.5 * (phi[i, j, k] + phi[i, j, k-1])
		return flux
