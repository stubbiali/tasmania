import pytest

import gridtools as gt


def test_first_order_upwind():
	k = gt.Index(axis=2)

	rho = gt.Equation()
	h   = gt.Equation()
	qr	= gt.Equation()
	vt  = gt.Equation()

	from tasmania.dynamics.sedimentation_flux import SedimentationFlux
	fluxer = SedimentationFlux.factory('first_order_upwind')

	flux = fluxer(k, rho, h, qr, vt)

	assert flux.get_name() == 'tmp_dfdz'


if __name__ == '__main__':
	pytest.main([__file__])