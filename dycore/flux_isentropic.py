"""
Numerical fluxes for the three-dimensional moist isentropic dynamical core.
"""
import abc
import numpy

import gridtools as gt

class FluxIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes for computing the numerical fluxes for 
	the three-dimensional isentropic dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, imoist):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._imoist = imoist

	def get_fluxes(self, i, j, k, dt, s, u, v, mtg, U, V, Qv = None, Qc = None, Qr = None):
		"""
		The entry-point method returning the :class:`gridtools.Equation`\s representing the :math:`x`- and 
		:math:`y`-fluxes for all the conservative model variables.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water.

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
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass of water vapour.
		flux_Qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass of water vapour.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass of cloud water.
		flux_Qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass of cloud water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the mass of precipitation water.
		flux_Qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the mass of precipitation water.
		"""
		self._compute_fluxes(i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr)
		if self._imoist:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y, \
				   self._flux_Qv_x, self._flux_Qv_y, self._flux_Qc_x, self._flux_Qc_y, self._flux_Qr_x, self._flux_Qr_y
		else:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y

	@staticmethod
	def factory(scheme, grid, imoist):
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
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if scheme == 'upwind':
			return FluxIsentropicUpwind(grid, imoist)
		elif scheme == 'centered':
			return FluxIsentropicCentered(grid, imoist)
		else:
			return FluxIsentropicMacCormack(grid, imoist)

	@abc.abstractmethod
	def _compute_fluxes(self, i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr):
		"""
		Method computing the :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s are then set as instance attributes.
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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : obj
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.
		"""


class FluxIsentropicUpwind(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the upwind scheme applied to the governing equations
	in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, imoist):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, imoist)
		self.nb = 1

	def _compute_fluxes(self, i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr):
		"""
		Method computing the upwind :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s are then set as instance attributes.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : obj
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.
		"""
		self._flux_s_x = self._get_upwind_flux_x(i, j, k, u, s)
		self._flux_s_y = self._get_upwind_flux_y(i, j, k, v, s)
		self._flux_U_x = self._get_upwind_flux_x(i, j, k, u, U)
		self._flux_U_y = self._get_upwind_flux_y(i, j, k, v, U)
		self._flux_V_x = self._get_upwind_flux_x(i, j, k, u, V)
		self._flux_V_y = self._get_upwind_flux_y(i, j, k, v, V)
		
		if self._imoist:
			self._flux_Qv_x = self._get_upwind_flux_x(i, j, k, u, Qv)
			self._flux_Qv_y = self._get_upwind_flux_y(i, j, k, v, Qv)
			self._flux_Qc_x = self._get_upwind_flux_x(i, j, k, u, Qc)
			self._flux_Qc_y = self._get_upwind_flux_y(i, j, k, v, Qc)
			self._flux_Qr_x = self._get_upwind_flux_x(i, j, k, u, Qr)
			self._flux_Qr_y = self._get_upwind_flux_y(i, j, k, v, Qr)

	def _get_upwind_flux_x(self, i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for a generic 
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
			:class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for :math:`\phi`
		"""
		# Note: by default, a GT4Py's Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = u[i, j, k] * ((u[i, j, k] > 0.) * phi[i-1, j, k] + 
									  (u[i, j, k] < 0.) * phi[  i, j, k])

		return flux

	def _get_upwind_flux_y(self, i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for a generic 
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
			:class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for :math:`\phi`
		"""
		# Note: by default, a GT4Py's Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = v[i, j, k] * ((v[i, j, k] > 0.) * phi[i, j-1, k] +
									  (v[i, j, k] < 0.) * phi[i,   j, k])

		return flux


class FluxIsentropicCentered(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the centered scheme applied to the governing equations
	in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, imoist):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, imoist)
		self.nb = 1

	def _compute_fluxes(self, i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr):
		"""
		Method computing the centered :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s are then set as instance attributes.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : obj
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.
		"""
		# Compute flux for the isentropic density and the momentums
		self._flux_s_x = self._get_centered_flux_x(i, j, k, u, s)
		self._flux_s_y = self._get_centered_flux_y(i, j, k, v, s)
		self._flux_U_x = self._get_centered_flux_x(i, j, k, u, U)
		self._flux_U_y = self._get_centered_flux_y(i, j, k, v, U)
		self._flux_V_x = self._get_centered_flux_x(i, j, k, u, V)
		self._flux_V_y = self._get_centered_flux_y(i, j, k, v, V)
		
		# Compute flux for the water constituents
		if self._imoist:
			self._flux_Qv_x = self._get_centered_flux_x(i, j, k, u, Qv)
			self._flux_Qv_y = self._get_centered_flux_y(i, j, k, v, Qv)
			self._flux_Qc_x = self._get_centered_flux_x(i, j, k, u, Qc)
			self._flux_Qc_y = self._get_centered_flux_y(i, j, k, v, Qc)
			self._flux_Qr_x = self._get_centered_flux_x(i, j, k, u, Qr)
			self._flux_Qr_y = self._get_centered_flux_y(i, j, k, v, Qr)

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
		flux[i, j, k] = u[i, j, k] * 0.5 * (phi[i-1, j, k] + phi[i, j, k])
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
		flux[i, j, k] = v[i, j, k] * 0.5 * (phi[i, j-1, k] + phi[i, j, k])
		return flux


class FluxIsentropicMacCormack(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the MacCormack scheme applied to the
	governing equations in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, imoist):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, imoist)
		self.nb = 2

	def _compute_fluxes(self, i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr):
		"""
		Method computing the MacCormack :class:`gridtools.Equation`\s representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s are then set as instance attributes.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : obj
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.
		"""
		# Diagnose unstaggered velocities
		u_unstg = gt.Equation()
		u_unstg[i, j, k] = 0. * (u[i, j, k] + u[i+1, j, k]) + U[i, j, k] / s[i, j, k]
		v_unstg = gt.Equation()
		v_unstg[i, j, k] = 0. * (v[i, j, k] + v[i, j+1, k]) + V[i, j, k] / s[i, j, k]

		# Compute predicted values
		s_pred = self._get_maccormack_predicted_value_density(i, j, k, dt, s, U, V)
		U_pred = self._get_maccormack_predicted_value_momentum_x(i, j, k, dt, s, u_unstg, v_unstg, mtg, U)
		V_pred = self._get_maccormack_predicted_value_momentum_y(i, j, k, dt, s, u_unstg, v_unstg, mtg, V)
		if self._imoist:
			Qv_pred = self._get_maccormack_predicted_value_constituent(i, j, k, dt, u_unstg, v_unstg, Qv)
			Qc_pred = self._get_maccormack_predicted_value_constituent(i, j, k, dt, u_unstg, v_unstg, Qc)
			Qr_pred = self._get_maccormack_predicted_value_constituent(i, j, k, dt, u_unstg, v_unstg, Qr)
		
		# Diagnose predicted values for the velocities
		u_unstg_pred = self._get_velocity(i, j, k, s_pred, U_pred)
		v_unstg_pred = self._get_velocity(i, j, k, s_pred, V_pred)

		# Get the fluxes
		self._flux_s_x = self._get_maccormack_flux_x_density(i, j, k, U, U_pred)
		self._flux_s_y = self._get_maccormack_flux_y_density(i, j, k, V, V_pred)
		self._flux_U_x = self._get_maccormack_flux_x(i, j, k, u_unstg, U, u_unstg_pred, U_pred)
		self._flux_U_y = self._get_maccormack_flux_y(i, j, k, v_unstg, U, v_unstg_pred, U_pred)
		self._flux_V_x = self._get_maccormack_flux_x(i, j, k, u_unstg, V, u_unstg_pred, V_pred)
		self._flux_V_y = self._get_maccormack_flux_y(i, j, k, v_unstg, V, v_unstg_pred, V_pred)
		if self._imoist:
			self._flux_Qv_x = self._get_maccormack_flux_x(i, j, k, u_unstg, Qv, u_unstg_pred, Qv_pred)
			self._flux_Qv_y = self._get_maccormack_flux_y(i, j, k, v_unstg, Qv, v_unstg_pred, Qv_pred)
			self._flux_Qc_x = self._get_maccormack_flux_x(i, j, k, u_unstg, Qc, u_unstg_pred, Qc_pred)
			self._flux_Qc_y = self._get_maccormack_flux_y(i, j, k, v_unstg, Qc, v_unstg_pred, Qc_pred)
			self._flux_Qr_x = self._get_maccormack_flux_x(i, j, k, u_unstg, Qr, u_unstg_pred, Qr_pred)
			self._flux_Qr_y = self._get_maccormack_flux_y(i, j, k, v_unstg, Qr, v_unstg_pred, Qr_pred)

	def _get_maccormack_predicted_value_density(self, i, j, k, dt, s, U, V):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the isentropic density.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the isentropic density.
		"""
		s_pred = gt.Equation()
		s_pred[i, j, k] = s[i, j, k] - dt * ((U[i+1, j, k] - U[i, j, k]) / self._grid.dx + 
											 (V[i, j+1, k] - V[i, j, k]) / self._grid.dy)
		return s_pred	

	def _get_maccormack_predicted_value_momentum_x(self, i, j, k, dt, s, u_unstg, v_unstg, mtg, U):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.
		"""
		U_pred = gt.Equation()
		U_pred[i, j, k] = U[i, j, k] - dt * ((u_unstg[i+1, j, k] * U[i+1, j, k] - 
											  u_unstg[  i, j, k] * U[  i, j, k]) / self._grid.dx + 
										 	 (v_unstg[i, j+1, k] * U[i, j+1, k] - 
											  v_unstg[i,   j, k] * U[i,   j, k]) / self._grid.dy +
											 s[i, j, k] * (mtg[i+1, j, k] - mtg[i, j, k]) / self._grid.dx)
		return U_pred	

	def _get_maccormack_predicted_value_momentum_y(self, i, j, k, dt, s, u_unstg, v_unstg, mtg, V):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.

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
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.
		"""
		V_pred = gt.Equation()
		V_pred[i, j, k] = V[i, j, k] - dt * ((u_unstg[i+1, j, k] * V[i+1, j, k] - 
											  u_unstg[  i, j, k] * V[  i, j, k]) / self._grid.dx + 
										 	 (v_unstg[i, j+1, k] * V[i, j+1, k] - 
											  v_unstg[i,   j, k] * V[i,   j, k]) / self._grid.dy +
											 s[i, j, k] * (mtg[i, j+1, k] - mtg[i, j, k]) / self._grid.dy)
		return V_pred	

	def _get_maccormack_predicted_value_constituent(self, i, j, k, dt, u_unstg, v_unstg, Q):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for mass of a generic water constituent :math:`Q`.

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
		u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		Q : obj
			:class:`gridtools.Equation` representing the mass of a generic water constituent :math:`Q`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for :math:`Q`.
		"""
		Q_name = Q.get_name()
		Q_pred_name = Q_name + '_pred'
		Q_pred = gt.Equation(name = Q_pred_name)
		Q_pred[i, j, k] = Q[i, j, k] - dt * ((u_unstg[i+1, j, k] * Q[i+1, j, k] -
											  u_unstg[  i, j, k] * Q[  i, j, k]) / self._grid.dx + 
										 	 (v_unstg[i, j+1, k] * Q[i, j+1, k] - 
											  v_unstg[i,   j, k] * Q[i,   j, k]) / self._grid.dy)
		return Q_pred	

	def _get_velocity(self, i, j, k, s, mnt):
		"""
		Get the :class:`gridtools.Equation` representing an unstaggered water component.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		mnt : obj
			:class:`gridtools.Equation` representing either the :math:`x`- or the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the diagnosed unstaggered velocity component.
		"""
		vel_name = mnt.get_name().lower()
		vel = gt.Equation(name = vel_name)
		vel[i, j, k] = mnt[i, j, k] / s[i, j, k]
		return vel

	def _get_maccormack_flux_x(self, i, j, k, u_unstg, phi, u_unstg_p, phi_p):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity at the current time.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		u_unstg_p : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`x`-velocity.
		phi_p : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (u_unstg[i, j, k] * phi[i, j, k] + u_unstg_p[i-1, j, k] * phi_p[i-1, j, k])
		return flux

	def _get_maccormack_flux_x_density(self, i, j, k, U, U_p):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for the
		isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		U_p : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for the isentropic density.
		"""
		flux_s_x = gt.Equation()
		flux_s_x[i, j, k] = 0.5 * (U[i, j, k] + U_p[i-1, j, k])
		return flux_s_x

	def _get_maccormack_flux_y(self, i, j, k, v_unstg, phi, v_unstg_p, phi_p):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity at the current time.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		v_unstg_p : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`y`-velocity.
		phi_p : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (v_unstg[i, j, k] * phi[i, j, k] + v_unstg_p[i, j-1, k] * phi_p[i, j-1, k])
		return flux

	def _get_maccormack_flux_y_density(self, i, j, k, V, V_p):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for the
		isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		V_p : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for the isentropic density.
		"""
		flux_s_y = gt.Equation()
		flux_s_y[i, j, k] = 0.5 * (V[i, j, k] + V_p[i, j-1, k])
		return flux_s_y
