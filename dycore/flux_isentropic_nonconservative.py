import abc

class FluxIsentropicNonconservative:
	"""
	Abstract base class whose derived classes implement different schemes to compute the numerical fluxes for 
	the three-dimensional isentropic dynamical core. The nonconservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

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
		self._grid = grid
		self._moist_on = moist_on

	def get_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, 
							  in_qv = None, in_qc = None, in_qr = None):
		"""
		Method returning the :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes 
		for all the prognostic model variables.

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
		in_qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of water vapour.
		in_qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
		in_qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density.
		flux_u_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`x`-velocity.
		flux_u_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`x`-velocity.
		flux_v_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`y`-velocity.
		flux_v_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`y`-velocity.
		flux_qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass fraction of water vapor.
		flux_qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass fraction of water vapor.
		flux_qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass fraction of cloud liquid water.
		flux_qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass fraction of cloud liquid water.
		flux_qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass fraction of precipitation water.
		flux_qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass fraction of precipitation water.
		"""
		self._compute_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_qv, in_qc, in_qr)
		if self._moist_on:
			return self._flux_s_x,  self._flux_s_y,  \
				   self._flux_u_x,  self._flux_u_y,  \
				   self._flux_v_x,  self._flux_v_y,  \
				   self._flux_qv_x, self._flux_qv_y, \
				   self._flux_qc_x, self._flux_qc_y, \
				   self._flux_qr_x, self._flux_qr_y
		else:
			return self._flux_s_x, self._flux_s_y, \
				   self._flux_u_x, self._flux_u_y, \
				   self._flux_v_x, self._flux_v_y

	def get_vertical_fluxes(self, i, j, k, dt, in_w, 
							in_s, in_s_prv, 
							in_u, in_u_prv, 
							in_v, in_v_prv, 
							in_qv = None, in_qv_prv = None, 
							in_qc = None, in_qc_prv = None,	
							in_qr = None, in_qr_prv = None):
		"""
		Method returning the :class:`gridtools.Equation`\s representing the :math:`\\theta`-flux for all 
		the prognostic model variables.

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
			the change over time in potential temperature.
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
		in_qv : `obj`, optional
			:class:`gridtools.Equation` representing the current mass fraction of water vapor.
		in_qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of water vapor, 
			i.e., the mass fraction of water vapor stepped disregarding the vertical advection.
		in_qc : `obj`, optional
			:class:`gridtools.Equation` representing the current mass fraction of cloud liquid water.
		in_qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of cloud liquid water, 
			i.e., the mass fraction of cloud liquid water stepped disregarding the vertical advection.
		in_qr : `obj`, optional
			:class:`gridtools.Equation` representing the current mass fraction of precipitation water.
		in_qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of precipitation water, 
			i.e., the mass fraction of precipitation water stepped disregarding the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the isentropic density.
		flux_u_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the :math:`x`-velocity.
		flux_v_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the :math:`y`-velocity.
		flux_qv_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the mass fraction of water vapor.
		flux_qc_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the mass fraction of cloud liquid water.
		flux_qr_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux for the mass fraction of precipitation water.
		"""
		self._compute_vertical_fluxes(i, j, k, dt, in_w, in_s, in_s_prv, in_u, in_u_prv, in_v, in_v_prv, 
									  in_qv, in_qv_prv, in_qc, in_qc_prv, in_qr, in_qr_prv)
		if self._moist_on:
			return self._flux_s_z, self._flux_u_z, self._flux_v_z, self._flux_qv_z, self._flux_qc_z, self._flux_qr_z
		else:
			return self._flux_s_z, self._flux_u_z, self._flux_v_z

	@staticmethod
	def factory(scheme, grid, moist_on):
		"""
		Static method which returns an instance of the derived class implementing the numerical scheme
		specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

			* 'centered', for a second-order centered scheme.

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if scheme == 'centered':
			from tasmania.dycore.flux_isentropic_nonconservative_centered import FluxIsentropicNonconservativeCentered
			return FluxIsentropicNonconservativeCentered(grid, moist_on)

	@abc.abstractmethod
	def _compute_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_qv, in_qc, in_qr):
		"""
		Method computing the :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes for 
		all the prognostic variables. The :class:`gridtools.Equation`\s are then set as instance attributes.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

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

	@abc.abstractmethod
	def _compute_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
								 in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s representing the :math:`\\theta`-fluxes for all the 
		prognostic model variables. The :class:`gridtools.Equation`\s are then set as instance attributes.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

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
