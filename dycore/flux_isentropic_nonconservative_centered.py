import gridtools as gt
from tasmania.dycore.flux_isentropic_nonconservative import FluxIsentropicNonconservative

class FluxIsentropicNonconservativeCentered(FluxIsentropicNonconservative):
	"""
	Class which inherits :class:`~dycore.flux_isentropic.FluxIsentropicNonconservative` to implement a 
	centered scheme to compute the numerical fluxes for the prognostic model variables. 
	The nonconservative form of the governing equations, expressed using isentropic coordinates, is used.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	order : int
		Order of accuracy.
	"""
	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def _compute_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_qv, in_qc, in_qr):
		"""
		Method computing the :class:`gridtools.Equation`~s representing the :math:`x`- and :math:`y`-fluxes for 
		all the prognostic variables. The :class:`gridtools.Equation`~s are then set as instance attributes.

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
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapour.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.
		"""
		# Compute the fluxes for the isentropic density and the velocity components
		self._flux_s_x = self._get_centered_flux_x_s(i, j, k, in_u, in_s)
		self._flux_s_y = self._get_centered_flux_y_s(i, j, k, in_v, in_s)
		self._flux_u_x = self._get_centered_flux_x_u(i, j, k, in_u)
		self._flux_u_y = self._get_centered_flux_y_unstg(i, j, k, in_u)
		self._flux_v_x = self._get_centered_flux_x_unstg(i, j, k, in_v)
		self._flux_v_y = self._get_centered_flux_y_v(i, j, k, in_v)

		if self._moist_on:
			# Compute the fluxes for the water constituents
			self._flux_qv_x = self._get_centered_flux_x_unstg(i, j, k, in_qv)
			self._flux_qv_y = self._get_centered_flux_y_unstg(i, j, k, in_qv)
			self._flux_qc_x = self._get_centered_flux_x_unstg(i, j, k, in_qc)
			self._flux_qc_y = self._get_centered_flux_y_unstg(i, j, k, in_qc)
			self._flux_qr_x = self._get_centered_flux_x_unstg(i, j, k, in_qr)
			self._flux_qr_y = self._get_centered_flux_y_unstg(i, j, k, in_qr)

	def _compute_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
								 in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`~s representing the :math:`\\theta`-fluxes for all the 
		prognostic model variables. The :class:`gridtools.Equation`~s are then set as instance attributes.

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
		in_w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., 
			the change over time of potential temperature.
		in_s : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., 
			the isentropic density stepped disregarding the vertical advection.
		in_u : obj
			:class:`gridtools.Equation` representing the current :math:`x`-velocity.
		in_u_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-velocity, i.e., 
			the :math:`x`-velocity stepped disregarding the vertical advection.
		in_v : obj
			:class:`gridtools.Equation` representing the current :math:`y`-velocity.
		in_v_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-velocity, i.e., 
			the :math:`y`-velocity stepped disregarding the vertical advection.
		in_qv : obj
			:class:`gridtools.Equation` representing the current mass fraction of water vapor.
		in_qv_prv : obj
			:class:`gridtools.Equation` representing the provisional mass fraction of water vapor, 
			i.e., the mass fraction of water vapor stepped disregarding the vertical advection.
		in_qc : obj			
			:class:`gridtools.Equation` representing the current mass fraction of cloud liquid water.
		in_qc_prv : obj
			:class:`gridtools.Equation` representing the provisional mass fraction of cloud liquid water, 
			i.e., the mass fraction of cloud liquid water stepped disregarding the vertical advection.
		in_qr : obj
			:class:`gridtools.Equation` representing the current mass fraction of precipitation water.
		in_qr_prv : obj
			:class:`gridtools.Equation` representing the provisional mass fraction of precipitation water, 
			i.e., the mass fraction of precipitation water stepped disregarding the vertical advection.
		"""
		### TODO ###

	def _get_centered_flux_x_s(self, i, j, k, in_u, in_s):
		"""
		Get the :math:`x`-flux for the isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density.
		"""
		flux_s_x = gt.Equation()
		flux_s_x[i, j, k] = 0.25 * (in_u[  i, j, k] + in_u[i+1, j, k]) * in_s[  i, j, k] + \
							0.25 * (in_u[i-1, j, k] + in_u[  i, j, k]) * in_s[i-1, j, k]
		return flux_s_x

	def _get_centered_flux_x_u(self, i, j, k, in_u):
		"""
		Get the :math:`x`-flux for the :math:`x`-velocity.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`x`-velocity.
		"""
		flux_u_x = gt.Equation()
		flux_u_x[i, j, k] = 0.5 * (in_u[i, j, k] + in_u[i+1, j, k])
		return flux_u_x

	def _get_centered_flux_x_unstg(self, i, j, k, in_phi):
		"""
		Get the :math:`x`-flux for a generic :math:`x`-unstaggered field.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_phi : obj
			:class:`gridtools.Equation` representing the advected field.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`x`-flux for the advected field.
		"""
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (in_phi[i-1, j, k] + in_phi[i, j, k])
		return flux

	def _get_centered_flux_y_s(self, i, j, k, in_v, in_s):
		"""
		Get the :math:`y`-flux for the isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density.
		"""
		flux_s_y = gt.Equation()
		flux_s_y[i, j, k] = 0.25 * (in_v[i,   j, k] + in_v[i, j+1, k]) * in_s[i,   j, k] + \
							0.25 * (in_v[i, j-1, k] + in_v[i,   j, k]) * in_s[i, j-1, k]
		return flux_s_y

	def _get_centered_flux_y_v(self, i, j, k, in_v):
		"""
		Get the :math:`y`-flux for the :math:`y`-velocity.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`y`-velocity.
		"""
		flux_v_y = gt.Equation()
		flux_v_y[i, j, k] = 0.5 * (in_v[i, j, k] + in_v[i, j+1, k])
		return flux_v_y

	def _get_centered_flux_y_unstg(self, i, j, k, in_phi):
		"""
		Get the :math:`y`-flux for a generic :math:`y`-unstaggered field.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_phi : obj
			:class:`gridtools.Equation` representing the advected field.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the :math:`y`-flux for the advected field.
		"""
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (in_phi[i, j-1, k] + in_phi[i, j, k])
		return flux
