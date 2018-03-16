import abc

class FluxIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes for computing the numerical fluxes for 
	the three-dimensional isentropic dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

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
		self._grid = grid
		self._moist_on = moist_on

	def get_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, 
							  Qv_now = None, Qc_now = None, Qr_now = None):
		"""
		Method returning the :class:`gridtools.Equation`\s_now representing the :math:`x`- and :math:`y`-fluxes 
		for all the conservative model variables.

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
		Qv_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density.
		flux_U_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`x`-momentum.
		flux_U_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`x`-momentum.
		flux_V_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`y`-momentum.
		flux_V_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`y`-momentum.
		flux_Qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of water vapour.
		flux_Qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of water vapour.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of cloud water.
		flux_Qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of cloud water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of precipitation water.
		flux_Qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of precipitation water.
		"""
		self._compute_horizontal_fluxes(i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now)
		if self._moist_on:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y, \
				   self._flux_Qv_x, self._flux_Qv_y, self._flux_Qc_x, self._flux_Qc_y, self._flux_Qr_x, self._flux_Qr_y
		else:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y

	def get_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
							Qv_now = None, Qv_prv = None, Qc_now = None, Qc_prv = None, Qr_now = None, Qr_prv = None):
		"""
		Method returning the :class:`gridtools.Equation`\s_now representing the :math:`z`-flux for all the conservative 
		model variables.

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
		Qv_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density.
		flux_U_x : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`x`-momentum.
		flux_V_x : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`y`-momentum.
		flux_Qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of water vapour.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of cloud water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of precipitation water.
		"""
		self._compute_vertical_fluxes(i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
									  Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv)
		if self._moist_on:
			return self._flux_s_z, self._flux_U_z, self._flux_V_z, self._flux_Qv_z, self._flux_Qc_z, self._flux_Qr_z
		else:
			return self._flux_s_z, self._flux_U_z, self._flux_V_z

	@staticmethod
	def factory(scheme, grid, moist_on):
		"""
		Static method which returns an instance of the derived class implementing the numerical scheme
		specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

			* 'upwind', for the upwind scheme;
			* 'centered', for a second-order centered scheme;
			* 'maccormack', for the MacCormack scheme.

		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if scheme == 'upwind':
			from tasmania.dycore.flux_isentropic_upwind import FluxIsentropicUpwind
			return FluxIsentropicUpwind(grid, moist_on)
		elif scheme == 'centered':
			from tasmania.dycore.flux_isentropic_centered import FluxIsentropicCentered
			return FluxIsentropicCentered(grid, moist_on)
		else:
			from tasmania.dycore.flux_isentropic_maccormack import FluxIsentropicMacCormack
			return FluxIsentropicMacCormack(grid, moist_on)

	@abc.abstractmethod
	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.
		As this method is marked as abstract, the implementation is delegated to the derived classes.

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

	@abc.abstractmethod
	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.
		As this method is marked as abstract, the implementation is delegated to the derived classes.

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
