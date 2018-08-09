"""
Classes:
	_{Upwind, Centered, MacCormack}IsentropicHorizontalFlux
	_{Upwind, Centered, MacCormack}IsentropicVerticalFlux
	_{Centered}NonconservativeIsentropicHorizontalFlux
	_{Centered}NonconservativeIsentropicVerticalFlux
"""
from tasmania.dynamics.isentropic_fluxes import IsentropicHorizontalFlux, \
												IsentropicVerticalFlux, \
												NonconservativeIsentropicHorizontalFlux, \
												NonconservativeIsentropicVerticalFlux

import gridtools as gt


class _UpwindIsentropicHorizontalFlux(IsentropicHorizontalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicHorizontalFlux`
	to implement the upwind scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 1

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`qv_tnd`, :data:`qc_tnd`, and :data:`qr_tnd` are not
		actually used, yet they appear as default arguments for compliancy
		with the class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = self._get_upwind_flux_x(i, j, k, u, s)
		flux_s_y  = self._get_upwind_flux_y(i, j, k, v, s)
		flux_su_x = self._get_upwind_flux_x(i, j, k, u, su)
		flux_su_y = self._get_upwind_flux_y(i, j, k, v, su)
		flux_sv_x = self._get_upwind_flux_x(i, j, k, u, sv)
		flux_sv_y = self._get_upwind_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = self._get_upwind_flux_x(i, j, k, u, sqv)
			flux_sqv_y = self._get_upwind_flux_y(i, j, k, v, sqv)
			flux_sqc_x = self._get_upwind_flux_x(i, j, k, u, sqc)
			flux_sqc_y = self._get_upwind_flux_y(i, j, k, v, sqc)
			flux_sqr_x = self._get_upwind_flux_x(i, j, k, u, sqr)
			flux_sqr_y = self._get_upwind_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list

	@staticmethod
	def _get_upwind_flux_x(i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind
		flux in :math:`x`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		# Note: by default, a GT4Py Equation instance is named with
		# the name used by the user to reference the object itself.
		# Here, this is likely to be dangerous as this method is called
		# on multiple instances of the Equation class. Hence, we explicitly
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = u[i+1, j, k] * \
						((u[i+1, j, k] > 0.) * phi[  i, j, k] +
						 (u[i+1, j, k] < 0.) * phi[i+1, j, k])

		return flux

	@staticmethod
	def _get_upwind_flux_y(i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind
		flux in :math:`y`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = v[i, j+1, k] * \
						((v[i, j+1, k] > 0.) * phi[i,   j, k] +
						 (v[i, j+1, k] < 0.) * phi[i, j+1, k])

		return flux


class _CenteredIsentropicHorizontalFlux(IsentropicHorizontalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicHorizontalFlux`
	to implement the centered scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`qv_tnd`, :data:`qc_tnd`, and :data:`qr_tnd` are not
		actually used, yet they appear as default arguments for compliancy
		with the class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = self._get_centered_flux_x(i, j, k, u, s)
		flux_s_y  = self._get_centered_flux_y(i, j, k, v, s)
		flux_su_x = self._get_centered_flux_x(i, j, k, u, su)
		flux_su_y = self._get_centered_flux_y(i, j, k, v, su)
		flux_sv_x = self._get_centered_flux_x(i, j, k, u, sv)
		flux_sv_y = self._get_centered_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = self._get_centered_flux_x(i, j, k, u, sqv)
			flux_sqv_y = self._get_centered_flux_y(i, j, k, v, sqv)
			flux_sqc_x = self._get_centered_flux_x(i, j, k, u, sqc)
			flux_sqc_y = self._get_centered_flux_y(i, j, k, v, sqc)
			flux_sqr_x = self._get_centered_flux_x(i, j, k, u, sqr)
			flux_sqr_y = self._get_centered_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list

	@staticmethod
	def _get_centered_flux_x(i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered
		flux in :math:`x`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = u[i+1, j, k] * 0.5 * (phi[i, j, k] + phi[i+1, j, k])

		return flux

	@staticmethod
	def _get_centered_flux_y(i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered
		flux in :math:`y`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = v[i, j+1, k] * 0.5 * (phi[i, j, k] + phi[i, j+1, k])

		return flux


class _MacCormackIsentropicHorizontalFlux(IsentropicHorizontalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicHorizontalFlux`
	to implement the MacCormack scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		# Diagnose the velocity components at the mass points
		u_unstg = gt.Equation()
		u_unstg[i, j, k] = su[i, j, k] / s[i, j, k]
		v_unstg = gt.Equation()
		v_unstg[i, j, k] = sv[i, j, k] / s[i, j, k]

		# Compute the predicted values for the isentropic density and the momenta
		s_prd = self._get_maccormack_horizontal_predicted_value_s(
			i, j, k, dt, s, su, sv)
		su_prd = self._get_maccormack_horizontal_predicted_value_su(
			i, j, k, dt, s, u_unstg, v_unstg, mtg, su)
		sv_prd = self._get_maccormack_horizontal_predicted_value_sv(
			i, j, k, dt, s, u_unstg, v_unstg, mtg, sv)

		if self._moist_on:
			# Compute the predicted values for the water constituents
			sqv_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqv, qr_tnd)
			sqc_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqc, qc_tnd)
			sqr_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqr, qv_tnd)

		# Diagnose the predicted values for the velocity components
		# at the mass points
		u_prd_unstg = gt.Equation()
		u_prd_unstg[i, j, k] = su_prd[i, j, k] / s_prd[i, j, k]
		v_prd_unstg = gt.Equation()
		v_prd_unstg[i, j, k] = sv_prd[i, j, k] / s_prd[i, j, k]

		# Compute the fluxes for the isentropic density and the momenta
		flux_s_x = self._get_maccormack_flux_x_s(i, j, k, su, su_prd)
		flux_s_y = self._get_maccormack_flux_y_s(i, j, k, sv, sv_prd)
		flux_su_x = self._get_maccormack_flux_x(i, j, k, u_unstg, su,
												u_prd_unstg, su_prd)
		flux_su_y = self._get_maccormack_flux_y(i, j, k, v_unstg, su,
												v_prd_unstg, su_prd)
		flux_sv_x = self._get_maccormack_flux_x(i, j, k, u_unstg, sv,
												u_prd_unstg, sv_prd)
		flux_sv_y = self._get_maccormack_flux_y(i, j, k, v_unstg, sv,
												v_prd_unstg, sv_prd)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute the fluxes for the water constituents
			flux_sqv_x = self._get_maccormack_flux_x(i, j, k, u_unstg, sqv,
													 u_prd_unstg, sqv_prd)
			flux_sqv_y = self._get_maccormack_flux_y(i, j, k, v_unstg, sqv,
													 v_prd_unstg, sqv_prd)
			flux_sqc_x = self._get_maccormack_flux_x(i, j, k, u_unstg, sqc,
													 u_prd_unstg, sqc_prd)
			flux_sqc_y = self._get_maccormack_flux_y(i, j, k, v_unstg, sqc,
													 v_prd_unstg, sqc_prd)
			flux_sqr_x = self._get_maccormack_flux_x(i, j, k, u_unstg, sqr,
													 u_prd_unstg, sqr_prd)
			flux_sqr_y = self._get_maccormack_flux_y(i, j, k, v_unstg, sqr,
													 v_prd_unstg, sqr_prd)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list

	def _get_maccormack_horizontal_predicted_value_s(self, i, j, k, dt, s, su, sv):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value
		for the isentropic density, computed without taking the vertical
		advection into account.
		"""
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		s_prd = gt.Equation()
		s_prd[i, j, k] = s[i, j, k] - \
						 dt * ((su[i+1, j, k] - su[i, j, k]) / dx +
							   (sv[i, j+1, k] - sv[i, j, k]) / dy)
		return s_prd

	def _get_maccormack_horizontal_predicted_value_su(self, i, j, k, dt, s,
													  u_unstg, v_unstg, mtg, su):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value
		for the :math:`x`-momentum, computed without taking the vertical
		advection into account.
		"""
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		su_prd = gt.Equation()
		su_prd[i, j, k] = su[i, j, k] - dt * \
						  ((u_unstg[i+1, j, k] * su[i+1, j, k] -
							u_unstg[  i, j, k] * su[  i, j, k]) / dx +
						   (v_unstg[i, j+1, k] * su[i, j+1, k] -
							v_unstg[i,   j, k] * su[i,   j, k]) / dy +
						   s[i, j, k] * (mtg[i+1, j, k] - mtg[i, j, k]) / dx)
		return su_prd

	def _get_maccormack_horizontal_predicted_value_sv(self, i, j, k, dt, s,
													  u_unstg, v_unstg, mtg, sv):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value
		for the :math:`y`-momentum, computed without taking the vertical
		advection into account.
		"""
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sv_prd = gt.Equation()
		sv_prd[i, j, k] = sv[i, j, k] - dt * \
						  ((u_unstg[i+1, j, k] * sv[i+1, j, k] -
							u_unstg[  i, j, k] * sv[  i, j, k]) / dx +
						   (v_unstg[i, j+1, k] * sv[i, j+1, k] -
							v_unstg[i,   j, k] * sv[i,   j, k]) / dy +
						   s[i, j, k] * (mtg[i, j+1, k] - mtg[i, j, k]) / dy)
		return sv_prd

	def _get_maccormack_horizontal_predicted_value_sq(self, i, j, k, dt, s,
													  u_unstg, v_unstg, sq, sq_tnd):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value
		for the isentropic density of a generic water constituent, computed
		without taking the vertical advection into account.
		"""
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sq_name = sq.get_name()
		sq_prd_name = sq_name + '_prd'
		sq_prd = gt.Equation(name=sq_prd_name)

		if sq_tnd is None:
			sq_prd[i, j, k] = sq[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sq[i+1, j, k] -
								u_unstg[  i, j, k] * sq[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sq[i, j+1, k] -
								v_unstg[i,   j, k] * sq[i,   j, k]) / dy)
		else:
			sq_prd[i, j, k] = sq[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sq[i+1, j, k] -
								u_unstg[  i, j, k] * sq[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sq[i, j+1, k] -
								v_unstg[i,   j, k] * sq[i,   j, k]) / dy +
							   s[i, j, k] * sq_tnd[i, j, k])

		return sq_prd

	@staticmethod
	def _get_maccormack_flux_x(i, j, k, u_unstg, phi, u_prd_unstg, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack
		flux in :math:`x`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = 0.5 * (u_unstg[i+1, j, k] * phi[i+1, j, k] +
							   u_prd_unstg[i, j, k] * phi_prd[i, j, k])

		return flux

	@staticmethod
	def _get_maccormack_flux_x_s(i, j, k, su, su_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack
		flux in :math:`x`-direction for the isentropic density.
		"""
		flux_s_x = gt.Equation()
		flux_s_x[i, j, k] = 0.5 * (su[i+1, j, k] + su_prd[i, j, k])
		return flux_s_x

	@staticmethod
	def _get_maccormack_flux_y(i, j, k, v_unstg, phi, v_prd_unstg, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack
		flux in :math:`y`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = 0.5 * (v_unstg[i, j+1, k] * phi[i, j+1, k] +
							   v_prd_unstg[i, j, k] * phi_prd[i, j, k])

		return flux

	@staticmethod
	def _get_maccormack_flux_y_s(i, j, k, sv, sv_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack
		flux in :math:`y`-direction for the isentropic density.
		"""
		flux_s_y = gt.Equation()
		flux_s_y[i, j, k] = 0.5 * (sv[i, j+1, k] + sv_prd[i, j, k])
		return flux_s_y


class _FifthOrderUpwindIsentropicHorizontalFlux(IsentropicHorizontalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicHorizontalFlux`
	to implement the fifth-order upwind scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 3
		self.order = 5

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`qv_tnd`, :data:`qc_tnd`, and :data:`qr_tnd` are not
		actually used, yet they appear as default arguments for compliancy
		with the class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = self._get_fifth_order_upwind_flux_x(i, j, k, u, s)
		flux_s_y  = self._get_fifth_order_upwind_flux_y(i, j, k, v, s)
		flux_su_x = self._get_fifth_order_upwind_flux_x(i, j, k, u, su)
		flux_su_y = self._get_fifth_order_upwind_flux_y(i, j, k, v, su)
		flux_sv_x = self._get_fifth_order_upwind_flux_x(i, j, k, u, sv)
		flux_sv_y = self._get_fifth_order_upwind_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = self._get_fifth_order_upwind_flux_x(i, j, k, u, sqv)
			flux_sqv_y = self._get_fifth_order_upwind_flux_y(i, j, k, v, sqv)
			flux_sqc_x = self._get_fifth_order_upwind_flux_x(i, j, k, u, sqc)
			flux_sqc_y = self._get_fifth_order_upwind_flux_y(i, j, k, v, sqc)
			flux_sqr_x = self._get_fifth_order_upwind_flux_x(i, j, k, u, sqr)
			flux_sqr_y = self._get_fifth_order_upwind_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list

	@staticmethod
	def _get_fifth_order_upwind_flux_x(i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the fifth-order
		upwind flux in :math:`x`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'fifth_order_flux_' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux6 = __class__._get_sixth_order_centered_flux_x(i, j, k, u, phi)

		flux[i, j, k] = flux6[i, j, k] - \
						((u[i+1, j, k] > 0.) * u[i+1, j, k] -
						 (u[i+1, j, k] < 0.) * u[i+1, j, k]) / 60. * \
						(10. * (phi[i+1, j, k] - phi[i, j, k])
						 - 5. * (phi[i+2, j, k] - phi[i-1, j, k])
						 + (phi[i+3, j, k] - phi[i-2, j, k]))

		return flux

	@staticmethod
	def _get_sixth_order_centered_flux_x(i, j, k, u, phi):
		phi_name = phi.get_name()
		flux_name = 'sixth_order_flux' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = u[i+1, j, k] / 60. * \
						(37. * (phi[i+1, j, k] + phi[i, j, k])
						 - 8. * (phi[i+2, j, k] + phi[i-1, j, k])
						 + (phi[i+3, j, k] + phi[i-2, j, k]))

		return flux

	@staticmethod
	def _get_fifth_order_upwind_flux_y(i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the fifth-order
		upwind flux in :math:`y`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'fifth_order_flux_' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux6 = __class__._get_sixth_order_centered_flux_y(i, j, k, v, phi)

		flux[i, j, k] = flux6[i, j, k] - \
						((v[i, j+1, k] > 0.) * v[i, j+1, k] -
						 (v[i, j+1, k] < 0.) * v[i, j+1, k]) / 60. * \
						(10. * (phi[i, j+1, k] - phi[i, j, k])
						 - 5. * (phi[i, j+2, k] - phi[i, j-1, k])
						 + (phi[i, j+3, k] - phi[i, j-2, k]))

		return flux

	@staticmethod
	def _get_sixth_order_centered_flux_y(i, j, k, v, phi):
		phi_name = phi.get_name()
		flux_name = 'sixth_order_flux' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = v[i, j+1, k] / 60. * \
						(37. * (phi[i, j+1, k] + phi[i, j, k])
						 - 8. * (phi[i, j+2, k] + phi[i, j-1, k])
						 + (phi[i, j+3, k] + phi[i, j-2, k]))

		return flux


class _UpwindIsentropicVerticalFlux(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the upwind scheme to compute the vertical
	numerical fluxes for the governing equations expressed
	in conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 1

	def __call__(self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
				 sqv=None, sqv_prv=None, sqc=None,
				 sqc_prv=None, sqr=None, sqr_prv=None):
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k-1])

		# Compute flux for the isentropic density and the momenta
		flux_s_z  = self._get_upwind_flux_z(i, j, k, w_mid, s)
		flux_su_z = self._get_upwind_flux_z(i, j, k, w_mid, su)
		flux_sv_z = self._get_upwind_flux_z(i, j, k, w_mid, sv)

		# Initialize return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist_on:
			# Compute flux for the water constituents
			flux_sqv_z = self._get_upwind_flux_z(i, j, k, w_mid, sqv)
			flux_sqc_z = self._get_upwind_flux_z(i, j, k, w_mid, sqc)
			flux_sqr_z = self._get_upwind_flux_z(i, j, k, w_mid, sqr)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list

	@staticmethod
	def _get_upwind_flux_z(i, j, k, w_mid, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind
		flux in :math:`\\theta`-direction for a generic prognostic
		variable :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = w_mid[i, j, k] * \
						((w_mid[i, j, k] > 0.) * phi[i, j, k-1] +
						 (w_mid[i, j, k] < 0.) * phi[i, j,   k])

		return flux


class _CenteredIsentropicVerticalFlux(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the centered scheme to compute the vertical
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
				 sqv=None, sqv_prv=None, sqc=None,
				 sqc_prv=None, sqr=None, sqr_prv=None):
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k-1])

		# Compute flux for the isentropic density and the momenta
		flux_s_z  = self._get_centered_flux_z(i, j, k, w_mid, s)
		flux_su_z = self._get_centered_flux_z(i, j, k, w_mid, su)
		flux_sv_z = self._get_centered_flux_z(i, j, k, w_mid, sv)

		# Initialize return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist_on:
			# Compute flux for the water constituents
			flux_sqv_z = self._get_centered_flux_z(i, j, k, w_mid, sqv)
			flux_sqc_z = self._get_centered_flux_z(i, j, k, w_mid, sqc)
			flux_sqr_z = self._get_centered_flux_z(i, j, k, w_mid, sqr)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list

	@staticmethod
	def _get_centered_flux_z(i, j, k, w_mid, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered
		flux in :math:`\\theta`-direction for a generic prognostic
		variable :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = w_mid[i, j, k] * 0.5 * (phi[i, j, k-1] + phi[i, j, k])

		return flux


class _MacCormackIsentropicVerticalFlux(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the MacCormack scheme to compute the vertical
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
				 sqv=None, sqv_prv=None, sqc=None,
				 sqc_prv=None, sqr=None, sqr_prv=None):
		# Compute the predicted values for the isentropic density
		# and the momenta
		s_prd  = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, s, s_prv)
		su_prd = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, su, su_prv)
		sv_prd = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, sv, sv_prv)

		if self._moist_on:
			# Compute the predicted values for the water constituents
			sqv_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqv, sqv_prv)
			sqc_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqc, sqc_prv)
			sqr_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqr, sqr_prv)

		# Compute the flux for the isentropic density and the momenta
		flux_s_z  = self._get_maccormack_flux_z(i, j, k, w, s, s_prd)
		flux_su_z = self._get_maccormack_flux_z(i, j, k, w, su, su_prd)
		flux_sv_z = self._get_maccormack_flux_z(i, j, k, w, sv, sv_prd)

		# Initialize the return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist_on:
			# Compute the flux for the water constituents
			flux_sqv_z = self._get_maccormack_flux_z(i, j, k, w, sqv, sqv_prd)
			flux_sqc_z = self._get_maccormack_flux_z(i, j, k, w, sqc, sqc_prd)
			flux_sqr_z = self._get_maccormack_flux_z(i, j, k, w, sqr, sqr_prd)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list

	def _get_maccormack_vertical_predicted_value(self, i, j, k, dt, w, phi, phi_prv):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value
		for a generic conservative prognostic variable :math:`\phi`, computed
		taking only the vertical advection into account.
		"""
		phi_name = phi.get_name()
		phi_prd_name = phi_name + '_prd'
		phi_prd = gt.Equation(name=phi_prd_name)

		dz = self._grid.dz.values.item()
		phi_prd[i, j, k] = phi_prv[i, j, k] - dt * \
						   (w[i, j, k-1] * phi[i, j, k-1] -
							w[i, j,   k] * phi[i, j,   k]) / dz

		return phi_prd

	@staticmethod
	def _get_maccormack_flux_z(i, j, k, w, phi, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack
		flux in :math:`\\theta`-direction for a generic prognostic variable
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = 0.5 * (w[i, j, k-1] * phi[i, j, k-1] +
							   w[i, j, k] * phi_prd[i, j, k])

		return flux


class _CenteredNonconservativeIsentropicHorizontalFlux(NonconservativeIsentropicHorizontalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.NonconservativeIsentropicHorizontalFlux`
	to implement a centered scheme to compute the horizontal
	numerical fluxes for the prognostic model variables.
	The nonconservative form of the governing equations,
	expressed using isentropic coordinates, is used.

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
        	:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
        	the underlying grid.
        moist_on : bool
        	:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        """
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, s, u, v, mtg, qv=None, qc=None, qr=None):
		# Compute the fluxes for the isentropic density and
		# the velocity components
		flux_s_x = self._get_centered_flux_x_s(i, j, k, u, s)
		flux_s_y = self._get_centered_flux_y_s(i, j, k, v, s)
		flux_u_x = self._get_centered_flux_x_u(i, j, k, u)
		flux_u_y = self._get_centered_flux_y_unstg(i, j, k, u)
		flux_v_x = self._get_centered_flux_x_unstg(i, j, k, v)
		flux_v_y = self._get_centered_flux_y_v(i, j, k, v)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_u_x, flux_u_y,
					   flux_v_x, flux_v_y]

		if self._moist_on:
			# Compute the fluxes for the water constituents
			flux_qv_x = self._get_centered_flux_x_unstg(i, j, k, qv)
			flux_qv_y = self._get_centered_flux_y_unstg(i, j, k, qv)
			flux_qc_x = self._get_centered_flux_x_unstg(i, j, k, qc)
			flux_qc_y = self._get_centered_flux_y_unstg(i, j, k, qc)
			flux_qr_x = self._get_centered_flux_x_unstg(i, j, k, qr)
			flux_qr_y = self._get_centered_flux_y_unstg(i, j, k, qr)

			# Update the return list
			return_list += [flux_qv_x, flux_qv_y, flux_qc_x, flux_qc_y,
							flux_qr_x, flux_qr_y]

		return return_list

	@staticmethod
	def _get_centered_flux_x_s(i, j, k, u, s):
		"""
		Get the :math:`x`-flux for the isentropic density.
		"""
		flux_s_x = gt.Equation()
		flux_s_x[i, j, k] = 0.25 * (u[  i, j, k] + u[i+1, j, k]) * s[  i, j, k] + \
							0.25 * (u[i-1, j, k] + u[  i, j, k]) * s[i-1, j, k]
		return flux_s_x

	@staticmethod
	def _get_centered_flux_x_u(i, j, k, u):
		"""
		Get the :math:`x`-flux for the :math:`x`-velocity.
		"""
		flux_u_x = gt.Equation()
		flux_u_x[i, j, k] = 0.5 * (u[i, j, k] + u[i+1, j, k])
		return flux_u_x

	@staticmethod
	def _get_centered_flux_x_unstg(i, j, k, phi):
		"""
		Get the :math:`x`-flux for a generic :math:`x`-unstaggered field
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = 0.5 * (phi[i-1, j, k] + phi[i, j, k])

		return flux

	@staticmethod
	def _get_centered_flux_y_s(i, j, k, v, s):
		"""
		Get the :math:`y`-flux for the isentropic density.
		"""
		flux_s_y = gt.Equation()
		flux_s_y[i, j, k] = 0.25 * (v[i,   j, k] + v[i, j+1, k]) * s[i,   j, k] + \
							0.25 * (v[i, j-1, k] + v[i,   j, k]) * s[i, j-1, k]
		return flux_s_y

	@staticmethod
	def _get_centered_flux_y_v(i, j, k, v):
		"""
		Get the :math:`y`-flux for the :math:`y`-velocity.
		"""
		flux_v_y = gt.Equation()
		flux_v_y[i, j, k] = 0.5 * (v[i, j, k] + v[i, j+1, k])
		return flux_v_y

	@staticmethod
	def _get_centered_flux_y_unstg(i, j, k, phi):
		"""
		Get the :math:`y`-flux for a generic :math:`y`-unstaggered field
		:math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name=flux_name)

		flux[i, j, k] = 0.5 * (phi[i, j-1, k] + phi[i, j, k])

		return flux


class _CenteredNonconservativeIsentropicVerticalFlux(NonconservativeIsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.NonconservativeIsentropicVerticalFlux`
	to implement a centered scheme to compute the horizontal
	numerical fluxes for the prognostic model variables.
	The nonconservative form of the governing equations,
	expressed using isentropic coordinates, is used.

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
        	:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
        	the underlying grid.
        moist_on : bool
        	:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        """
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(self, i, j, k, dt, w, s, s_prv, u, u_prv, v, v_prv,
				 qv=None, qv_prv=None, qc=None, qc_prv=None, qr=None, qr_prv=None):
		raise NotImplementedError()
