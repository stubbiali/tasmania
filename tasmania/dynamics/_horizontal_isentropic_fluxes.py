"""
This module contains:
	Upwind(HorizontalIsentropicFlux)
	Centered(HorizontalIsentropicFlux)
	MacCormack(HorizontalIsentropicFlux)
	ThirdOrderUpwind(HorizontalIsentropicFlux)
	FifthOrderUpwind(HorizontalIsentropicFlux)

	get_upwind_flux_{x, y}
	get_centered_flux_{x, y}
	get_maccormack_flux_{x, y}
	get_maccormack_flux_{x, y}_s
	get_third_order_upwind_flux_{x, y}
	get_fourth_order_centered_flux_{x, y}
	get_fifth_order_upwind_flux_{x, y}
	get_sixth_order_centered_flux_{x, y}
"""
import gridtools as gt
from tasmania.dynamics.isentropic_fluxes import HorizontalIsentropicFlux


class Upwind(HorizontalIsentropicFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
	to implement the upwind scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 1

	@property
	def order(self):
		return 1

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 s_tnd=None, su_tnd=None, sv_tnd=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`s_tnd, `:data:`su_tnd`, :data:`sv_tnd`, :data:`qv_tnd`,
		:data:`qc_tnd`, and :data:`qr_tnd` are not actually used, yet
		they are retained as default arguments for compliancy with the
		class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_upwind_flux_x(i, j, k, u, s)
		flux_s_y  = get_upwind_flux_y(i, j, k, v, s)
		flux_su_x = get_upwind_flux_x(i, j, k, u, su)
		flux_su_y = get_upwind_flux_y(i, j, k, v, su)
		flux_sv_x = get_upwind_flux_x(i, j, k, u, sv)
		flux_sv_y = get_upwind_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = get_upwind_flux_x(i, j, k, u, sqv)
			flux_sqv_y = get_upwind_flux_y(i, j, k, v, sqv)
			flux_sqc_x = get_upwind_flux_x(i, j, k, u, sqc)
			flux_sqc_y = get_upwind_flux_y(i, j, k, v, sqc)
			flux_sqr_x = get_upwind_flux_x(i, j, k, u, sqr)
			flux_sqr_y = get_upwind_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list


def get_upwind_flux_x(i, j, k, u, phi):
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


def get_upwind_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = v[i, j+1, k] * \
					((v[i, j+1, k] > 0.) * phi[i,	j, k] +
					 (v[i, j+1, k] < 0.) * phi[i, j+1, k])

	return flux


class Centered(HorizontalIsentropicFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
	to implement the centered scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 1

	@property
	def order(self):
		return 2

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 s_tnd=None, su_tnd=None, sv_tnd=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`s_tnd, `:data:`su_tnd`, :data:`sv_tnd`, :data:`qv_tnd`,
		:data:`qc_tnd`, and :data:`qr_tnd` are not actually used, yet
		they are retained as default arguments for compliancy with the
		class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_centered_flux_x(i, j, k, u, s)
		flux_s_y  = get_centered_flux_y(i, j, k, v, s)
		flux_su_x = get_centered_flux_x(i, j, k, u, su)
		flux_su_y = get_centered_flux_y(i, j, k, v, su)
		flux_sv_x = get_centered_flux_x(i, j, k, u, sv)
		flux_sv_y = get_centered_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = get_centered_flux_x(i, j, k, u, sqv)
			flux_sqv_y = get_centered_flux_y(i, j, k, v, sqv)
			flux_sqc_x = get_centered_flux_x(i, j, k, u, sqc)
			flux_sqc_y = get_centered_flux_y(i, j, k, v, sqc)
			flux_sqr_x = get_centered_flux_x(i, j, k, u, sqr)
			flux_sqr_y = get_centered_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list


def get_centered_flux_x(i, j, k, u, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = u[i+1, j, k] * 0.5 * (phi[i, j, k] + phi[i+1, j, k])

	return flux


def get_centered_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = v[i, j+1, k] * 0.5 * (phi[i, j, k] + phi[i, j+1, k])

	return flux


class MacCormack(HorizontalIsentropicFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
	to implement the MacCormack scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 1

	@property
	def order(self):
		return 2

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 s_tnd=None, su_tnd=None, sv_tnd=None,
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
			i, j, k, dt, s, u_unstg, v_unstg, mtg, su, su_tnd)
		sv_prd = self._get_maccormack_horizontal_predicted_value_sv(
			i, j, k, dt, s, u_unstg, v_unstg, mtg, sv, sv_tnd)

		if self._moist_on:
			# Compute the predicted values for the water constituents
			sqv_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqv, qv_tnd)
			sqc_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqc, qc_tnd)
			sqr_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, k, dt, s, u_unstg, v_unstg, sqr, qr_tnd)

		# Diagnose the predicted values for the velocity components
		# at the mass points
		u_prd_unstg = gt.Equation()
		u_prd_unstg[i, j, k] = su_prd[i, j, k] / s_prd[i, j, k]
		v_prd_unstg = gt.Equation()
		v_prd_unstg[i, j, k] = sv_prd[i, j, k] / s_prd[i, j, k]

		# Compute the fluxes for the isentropic density and the momenta
		flux_s_x  = get_maccormack_flux_x_s(i, j, k, su, su_prd)
		flux_s_y  = get_maccormack_flux_y_s(i, j, k, sv, sv_prd)
		flux_su_x = get_maccormack_flux_x(i, j, k, u_unstg, su, u_prd_unstg, su_prd)
		flux_su_y = get_maccormack_flux_y(i, j, k, v_unstg, su, v_prd_unstg, su_prd)
		flux_sv_x = get_maccormack_flux_x(i, j, k, u_unstg, sv, u_prd_unstg, sv_prd)
		flux_sv_y = get_maccormack_flux_y(i, j, k, v_unstg, sv, v_prd_unstg, sv_prd)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute the fluxes for the water constituents
			flux_sqv_x = get_maccormack_flux_x(i, j, k, u_unstg, sqv, u_prd_unstg, sqv_prd)
			flux_sqv_y = get_maccormack_flux_y(i, j, k, v_unstg, sqv, v_prd_unstg, sqv_prd)
			flux_sqc_x = get_maccormack_flux_x(i, j, k, u_unstg, sqc, u_prd_unstg, sqc_prd)
			flux_sqc_y = get_maccormack_flux_y(i, j, k, v_unstg, sqc, v_prd_unstg, sqc_prd)
			flux_sqr_x = get_maccormack_flux_x(i, j, k, u_unstg, sqr, u_prd_unstg, sqr_prd)
			flux_sqr_y = get_maccormack_flux_y(i, j, k, v_unstg, sqr, v_prd_unstg, sqr_prd)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list

	def _get_maccormack_horizontal_predicted_value_s(self, i, j, k, dt, s, su, sv):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		s_prd = gt.Equation()
		s_prd[i, j, k] = s[i, j, k] - \
						 dt * ((su[i+1, j, k] - su[i, j, k]) / dx +
							   (sv[i, j+1, k] - sv[i, j, k]) / dy)
		return s_prd

	def _get_maccormack_horizontal_predicted_value_su(self, i, j, k, dt, s,
													  u_unstg, v_unstg, mtg, su, su_tnd):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		su_prd = gt.Equation()

		if su_tnd is None:
			su_prd[i, j, k] = su[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * su[i+1, j, k] -
								u_unstg[  i, j, k] * su[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * su[i, j+1, k] -
								v_unstg[i,	 j, k] * su[i,	 j, k]) / dy +
							   s[i, j, k] * (mtg[i+1, j, k] - mtg[i, j, k]) / dx)
		else:
			su_prd[i, j, k] = su[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * su[i+1, j, k] -
								u_unstg[  i, j, k] * su[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * su[i, j+1, k] -
								v_unstg[i,	 j, k] * su[i,	 j, k]) / dy +
							   s[i, j, k] * (mtg[i+1, j, k] - mtg[i, j, k]) / dx -
							   su_tnd[i, j, k])

		return su_prd

	def _get_maccormack_horizontal_predicted_value_sv(self, i, j, k, dt, s,
													  u_unstg, v_unstg, mtg, sv, sv_tnd):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sv_prd = gt.Equation()

		if sv_tnd is None:
			sv_prd[i, j, k] = sv[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sv[i+1, j, k] -
								u_unstg[  i, j, k] * sv[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sv[i, j+1, k] -
								v_unstg[i,	 j, k] * sv[i,	 j, k]) / dy +
							   s[i, j, k] * (mtg[i, j+1, k] - mtg[i, j, k]) / dy)
		else:
			sv_prd[i, j, k] = sv[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sv[i+1, j, k] -
								u_unstg[  i, j, k] * sv[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sv[i, j+1, k] -
								v_unstg[i,	 j, k] * sv[i,	 j, k]) / dy +
							   s[i, j, k] * (mtg[i, j+1, k] - mtg[i, j, k]) / dy -
							   sv_tnd[i, j, k])

		return sv_prd

	def _get_maccormack_horizontal_predicted_value_sq(self, i, j, k, dt, s,
													  u_unstg, v_unstg, sq, q_tnd):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sq_name = sq.get_name()
		sq_prd_name = sq_name + '_prd'
		sq_prd = gt.Equation(name=sq_prd_name)

		if q_tnd is None:
			sq_prd[i, j, k] = sq[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sq[i+1, j, k] -
								u_unstg[  i, j, k] * sq[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sq[i, j+1, k] -
								v_unstg[i,	 j, k] * sq[i,	 j, k]) / dy)
		else:
			sq_prd[i, j, k] = sq[i, j, k] - dt * \
							  ((u_unstg[i+1, j, k] * sq[i+1, j, k] -
								u_unstg[  i, j, k] * sq[  i, j, k]) / dx +
							   (v_unstg[i, j+1, k] * sq[i, j+1, k] -
								v_unstg[i,	 j, k] * sq[i,	 j, k]) / dy -
							   s[i, j, k] * q_tnd[i, j, k])

		return sq_prd


def get_maccormack_flux_x(i, j, k, u_unstg, phi, u_prd_unstg, phi_prd):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = 0.5 * (u_unstg[i+1, j, k] * phi[i+1, j, k] +
						   u_prd_unstg[i, j, k] * phi_prd[i, j, k])

	return flux


def get_maccormack_flux_x_s(i, j, k, su, su_prd):
	flux_s_x = gt.Equation()
	flux_s_x[i, j, k] = 0.5 * (su[i+1, j, k] + su_prd[i, j, k])
	return flux_s_x


def get_maccormack_flux_y(i, j, k, v_unstg, phi, v_prd_unstg, phi_prd):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = 0.5 * (v_unstg[i, j+1, k] * phi[i, j+1, k] +
						   v_prd_unstg[i, j, k] * phi_prd[i, j, k])

	return flux


def get_maccormack_flux_y_s(i, j, k, sv, sv_prd):
	flux_s_y = gt.Equation()
	flux_s_y[i, j, k] = 0.5 * (sv[i, j+1, k] + sv_prd[i, j, k])
	return flux_s_y


class ThirdOrderUpwind(HorizontalIsentropicFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
	to implement the third-order upwind scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 2

	@property
	def order(self):
		return 3

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 s_tnd=None, su_tnd=None, sv_tnd=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`s_tnd, `:data:`su_tnd`, :data:`sv_tnd`, :data:`qv_tnd`,
		:data:`qc_tnd`, and :data:`qr_tnd` are not actually used, yet
		they are retained as default arguments for compliancy with the
		class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_third_order_upwind_flux_x(i, j, k, u, s)
		flux_s_y  = get_third_order_upwind_flux_y(i, j, k, v, s)
		flux_su_x = get_third_order_upwind_flux_x(i, j, k, u, su)
		flux_su_y = get_third_order_upwind_flux_y(i, j, k, v, su)
		flux_sv_x = get_third_order_upwind_flux_x(i, j, k, u, sv)
		flux_sv_y = get_third_order_upwind_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = get_third_order_upwind_flux_x(i, j, k, u, sqv)
			flux_sqv_y = get_third_order_upwind_flux_y(i, j, k, v, sqv)
			flux_sqc_x = get_third_order_upwind_flux_x(i, j, k, u, sqc)
			flux_sqc_y = get_third_order_upwind_flux_y(i, j, k, v, sqc)
			flux_sqr_x = get_third_order_upwind_flux_x(i, j, k, u, sqr)
			flux_sqr_y = get_third_order_upwind_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list


def get_third_order_upwind_flux_x(i, j, k, u, phi):
	phi_name = phi.get_name()
	flux_name = 'third_order_flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux4 = get_fourth_order_centered_flux_x(i, j, k, u, phi)

	flux[i, j, k] = flux4[i, j, k] - \
					((u[i+1, j, k] > 0.) * u[i+1, j, k] -
					 (u[i+1, j, k] < 0.) * u[i+1, j, k]) / 12. * \
					(3. * (phi[i+1, j, k] - phi[  i, j, k]) -
					 	  (phi[i+2, j, k] - phi[i-1, j, k]))

	return flux


def get_fourth_order_centered_flux_x(i, j, k, u, phi):
	phi_name = phi.get_name()
	flux_name = 'fourth_order_flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = u[i+1, j, k] / 12. * \
					(7. * (phi[i+1, j, k] + phi[  i, j, k]) -
					 	  (phi[i+2, j, k] + phi[i-1, j, k]))

	return flux


def get_third_order_upwind_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'third_order_flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux4 = get_fourth_order_centered_flux_y(i, j, k, v, phi)

	flux[i, j, k] = flux4[i, j, k] - \
					((v[i, j+1, k] > 0.) * v[i, j+1, k] -
					 (v[i, j+1, k] < 0.) * v[i, j+1, k]) / 60. * \
					(3. * (phi[i, j+1, k] - phi[i,   j, k]) -
					 	  (phi[i, j+2, k] - phi[i, j-1, k]))

	return flux


def get_fourth_order_centered_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'fourth_order_flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = v[i, j+1, k] / 12. * \
					(7. * (phi[i, j+1, k] + phi[i, j, k]) -
					 	  (phi[i, j+2, k] + phi[i, j-1, k]))

	return flux


class FifthOrderUpwind(HorizontalIsentropicFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
	to implement the fifth-order upwind scheme to compute the horizontal
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 3

	@property
	def order(self):
		return 5

	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv,
				 sqv=None, sqc=None, sqr=None,
				 s_tnd=None, su_tnd=None, sv_tnd=None,
				 qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		Note
		----
		:data:`s_tnd, `:data:`su_tnd`, :data:`sv_tnd`, :data:`qv_tnd`,
		:data:`qc_tnd`, and :data:`qr_tnd` are not actually used, yet
		they are retained as default arguments for compliancy with the
		class hierarchy interface.
		"""
		# Compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_fifth_order_upwind_flux_x(i, j, k, u, s)
		flux_s_y  = get_fifth_order_upwind_flux_y(i, j, k, v, s)
		flux_su_x = get_fifth_order_upwind_flux_x(i, j, k, u, su)
		flux_su_y = get_fifth_order_upwind_flux_y(i, j, k, v, su)
		flux_sv_x = get_fifth_order_upwind_flux_x(i, j, k, u, sv)
		flux_sv_y = get_fifth_order_upwind_flux_y(i, j, k, v, sv)

		# Initialize the return list
		return_list = [flux_s_x, flux_s_y, flux_su_x, flux_su_y,
					   flux_sv_x, flux_sv_y]

		if self._moist_on:
			# Compute fluxes for the water constituents
			flux_sqv_x = get_fifth_order_upwind_flux_x(i, j, k, u, sqv)
			flux_sqv_y = get_fifth_order_upwind_flux_y(i, j, k, v, sqv)
			flux_sqc_x = get_fifth_order_upwind_flux_x(i, j, k, u, sqc)
			flux_sqc_y = get_fifth_order_upwind_flux_y(i, j, k, v, sqc)
			flux_sqr_x = get_fifth_order_upwind_flux_x(i, j, k, u, sqr)
			flux_sqr_y = get_fifth_order_upwind_flux_y(i, j, k, v, sqr)

			# Update the return list
			return_list += [flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y,
							flux_sqr_x, flux_sqr_y]

		return return_list


def get_fifth_order_upwind_flux_x(i, j, k, u, phi):
	phi_name = phi.get_name()
	flux_name = 'fifth_order_flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux6 = get_sixth_order_centered_flux_x(i, j, k, u, phi)

	flux[i, j, k] = flux6[i, j, k] - \
					((u[i+1, j, k] > 0.) * u[i+1, j, k] -
					 (u[i+1, j, k] < 0.) * u[i+1, j, k]) / 60. * \
					(10. * (phi[i+1, j, k] - phi[  i, j, k]) -
					  5. * (phi[i+2, j, k] - phi[i-1, j, k]) +
					 	   (phi[i+3, j, k] - phi[i-2, j, k]))

	return flux


def get_sixth_order_centered_flux_x(i, j, k, u, phi):
	phi_name = phi.get_name()
	flux_name = 'sixth_order_flux_' + phi_name + '_x'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = u[i+1, j, k] / 60. * \
					(37. * (phi[i+1, j, k] + phi[  i, j, k]) -
					  8. * (phi[i+2, j, k] + phi[i-1, j, k]) +
					       (phi[i+3, j, k] + phi[i-2, j, k]))

	return flux


def get_fifth_order_upwind_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'fifth_order_flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux6 = get_sixth_order_centered_flux_y(i, j, k, v, phi)

	flux[i, j, k] = flux6[i, j, k] - \
					((v[i, j+1, k] > 0.) * v[i, j+1, k] -
					 (v[i, j+1, k] < 0.) * v[i, j+1, k]) / 60. * \
					(10. * (phi[i, j+1, k] - phi[i,   j, k]) -
					  5. * (phi[i, j+2, k] - phi[i, j-1, k]) +
					 	   (phi[i, j+3, k] - phi[i, j-2, k]))

	return flux


def get_sixth_order_centered_flux_y(i, j, k, v, phi):
	phi_name = phi.get_name()
	flux_name = 'sixth_order_flux_' + phi_name + '_y'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = v[i, j+1, k] / 60. * \
					(37. * (phi[i, j+1, k] + phi[i, j, k]) -
					  8. * (phi[i, j+2, k] + phi[i, j-1, k]) +
					       (phi[i, j+3, k] + phi[i, j-2, k]))

	return flux
