import pytest

import gridtools as gt


def test_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc   	= gt.Equation()
	sqr    	= gt.Equation()
	u_tnd  	= gt.Equation()
	v_tnd  	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, u_tnd=u_tnd, v_tnd=v_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, sqv, sqc, sqr,
					u_tnd=u_tnd, v_tnd=v_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_centered_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	u_tnd  	= gt.Equation()
	v_tnd  	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('centered', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, u_tnd=u_tnd, v_tnd=v_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('centered', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, sqv, sqc, sqr,
					u_tnd=u_tnd, v_tnd=v_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_maccormack_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	u_tnd  	= gt.Equation()
	v_tnd  	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('maccormack', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, u_tnd=u_tnd, v_tnd=v_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('maccormack', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, sqv, sqc, sqr,
					u_tnd=u_tnd, v_tnd=v_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_third_order_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	u_tnd  	= gt.Equation()
	v_tnd  	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('third_order_upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, u_tnd=u_tnd, v_tnd=v_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'third_order_flux_s_x'
	assert fluxes[1].get_name() == 'third_order_flux_s_y'
	assert fluxes[2].get_name() == 'third_order_flux_su_x'
	assert fluxes[3].get_name() == 'third_order_flux_su_y'
	assert fluxes[4].get_name() == 'third_order_flux_sv_x'
	assert fluxes[5].get_name() == 'third_order_flux_sv_y'

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('third_order_upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, sqv, sqc, sqr,
					u_tnd=u_tnd, v_tnd=v_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'third_order_flux_s_x'
	assert fluxes[1].get_name()  == 'third_order_flux_s_y'
	assert fluxes[2].get_name()  == 'third_order_flux_su_x'
	assert fluxes[3].get_name()  == 'third_order_flux_su_y'
	assert fluxes[4].get_name()  == 'third_order_flux_sv_x'
	assert fluxes[5].get_name()  == 'third_order_flux_sv_y'
	assert fluxes[6].get_name()  == 'third_order_flux_sqv_x'
	assert fluxes[7].get_name()  == 'third_order_flux_sqv_y'
	assert fluxes[8].get_name()  == 'third_order_flux_sqc_x'
	assert fluxes[9].get_name()  == 'third_order_flux_sqc_y'
	assert fluxes[10].get_name() == 'third_order_flux_sqr_x'
	assert fluxes[11].get_name() == 'third_order_flux_sqr_y'


def test_fifth_order_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	u_tnd  	= gt.Equation()
	v_tnd  	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.dynamics.isentropic_fluxes import HorizontalHomogeneousIsentropicFlux

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('fifth_order_upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, u_tnd=u_tnd, v_tnd=v_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'fifth_order_flux_s_x'
	assert fluxes[1].get_name() == 'fifth_order_flux_s_y'
	assert fluxes[2].get_name() == 'fifth_order_flux_su_x'
	assert fluxes[3].get_name() == 'fifth_order_flux_su_y'
	assert fluxes[4].get_name() == 'fifth_order_flux_sv_x'
	assert fluxes[5].get_name() == 'fifth_order_flux_sv_y'

	fluxer = HorizontalHomogeneousIsentropicFlux.factory('fifth_order_upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, su, sv, sqv, sqc, sqr,
					u_tnd=u_tnd, v_tnd=v_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'fifth_order_flux_s_x'
	assert fluxes[1].get_name()  == 'fifth_order_flux_s_y'
	assert fluxes[2].get_name()  == 'fifth_order_flux_su_x'
	assert fluxes[3].get_name()  == 'fifth_order_flux_su_y'
	assert fluxes[4].get_name()  == 'fifth_order_flux_sv_x'
	assert fluxes[5].get_name()  == 'fifth_order_flux_sv_y'
	assert fluxes[6].get_name()  == 'fifth_order_flux_sqv_x'
	assert fluxes[7].get_name()  == 'fifth_order_flux_sqv_y'
	assert fluxes[8].get_name()  == 'fifth_order_flux_sqc_x'
	assert fluxes[9].get_name()  == 'fifth_order_flux_sqc_y'
	assert fluxes[10].get_name() == 'fifth_order_flux_sqr_x'
	assert fluxes[11].get_name() == 'fifth_order_flux_sqr_y'


if __name__ == '__main__':
	pytest.main([__file__])