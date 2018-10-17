import numpy as np
import pytest

import gridtools as gt
from tasmania.physics.isentropic_tendencies import NonconservativeIsentropicPressureGradient, \
												   ConservativeIsentropicPressureGradient


def test_nonconservative_relaxed_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = NonconservativeIsentropicPressureGradient(
		grid, 2, 'relaxed', backend=backend, dtype=dtype)
	pg4 = NonconservativeIsentropicPressureGradient(
		grid, 4, 'relaxed', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	mtg = state['montgomery_potential'].values

	u_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	u_tnd2[1:-1, 1:-1, :] = - (mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_velocity' in tendencies2.keys()
	assert np.allclose(u_tnd2, tendencies2['x_velocity'])

	v_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	v_tnd2[1:-1, 1:-1, :] = - (mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	assert 'y_velocity' in tendencies2.keys()
	assert np.allclose(v_tnd2, tendencies2['y_velocity'])

	u_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	u_tnd4[2:-2, 2:-2, :] = - (mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
						       8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]) / (12. * dx)
	assert 'x_velocity' in tendencies4.keys()
	assert np.allclose(u_tnd4, tendencies4['x_velocity'])

	v_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	v_tnd4[2:-2, 2:-2, :] = - (mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
							   8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]) / (12. * dy)
	assert 'y_velocity' in tendencies4.keys()
	assert np.allclose(v_tnd4, tendencies4['y_velocity'])


def test_nonconservative_periodic_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = NonconservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=backend, dtype=dtype)
	pg4 = NonconservativeIsentropicPressureGradient(
		grid, 4, 'periodic', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	mtg = state['montgomery_potential'].values
	_mtg2 = np.concatenate((mtg[-2:-1, :, :], mtg, mtg[1:2, :, :]), axis=0)
	mtg2  = np.concatenate((_mtg2[:, -2:-1, :], _mtg2, _mtg2[:, 1:2, :]), axis=1)
	_mtg4 = np.concatenate((mtg[-3:-1, :, :], mtg, mtg[1:3, :, :]), axis=0)
	mtg4  = np.concatenate((_mtg4[:, -3:-1, :], _mtg4, _mtg4[:, 1:3, :]), axis=1)

	u_tnd2 = - (mtg2[2:, 1:-1, :] - mtg2[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_velocity' in tendencies2.keys()
	assert np.allclose(u_tnd2, tendencies2['x_velocity'])

	v_tnd2 = - (mtg2[1:-1, 2:, :] - mtg2[1:-1, :-2, :]) / (2. * dy)
	assert 'y_velocity' in tendencies2.keys()
	assert np.allclose(v_tnd2, tendencies2['y_velocity'])

	u_tnd4 = - (mtg4[:-4, 2:-2, :] - 8. * mtg4[1:-3, 2:-2, :] +
			    8. * mtg4[3:-1, 2:-2, :] - mtg4[4:, 2:-2, :]) / (12. * dx)
	assert 'x_velocity' in tendencies4.keys()
	assert np.allclose(u_tnd4, tendencies4['x_velocity'])

	v_tnd4 = - (mtg4[2:-2, :-4, :] - 8. * mtg4[2:-2, 1:-3, :] +
			    8. * mtg4[2:-2, 3:-1, :] - mtg4[2:-2, 4:, :]) / (12. * dy)
	assert 'y_velocity' in tendencies4.keys()
	assert np.allclose(v_tnd4, tendencies4['y_velocity'])


def test_conservative_relaxed_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = ConservativeIsentropicPressureGradient(
		grid, 2, 'relaxed', backend=backend, dtype=dtype)
	pg4 = ConservativeIsentropicPressureGradient(
		grid, 4, 'relaxed', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	s   = state['air_isentropic_density'].values
	mtg = state['montgomery_potential'].values

	su_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	su_tnd2[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
							 (mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(su_tnd2, tendencies2['x_momentum_isentropic'])

	sv_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	sv_tnd2[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
							 (mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	assert 'y_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(sv_tnd2, tendencies2['y_momentum_isentropic'])

	su_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	su_tnd4[2:-2, 2:-2, :] = - s[2:-2, 2:-2, :] * \
							 (mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
							  8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]) / (12. * dx)
	assert 'x_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(su_tnd4, tendencies4['x_momentum_isentropic'])

	sv_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	sv_tnd4[2:-2, 2:-2, :] = -s[2:-2, 2:-2, :] * \
							 (mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
							  8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]) / (12. * dy)
	assert 'y_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(sv_tnd4, tendencies4['y_momentum_isentropic'])


def test_conservative_periodic_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=backend, dtype=dtype)
	pg4 = ConservativeIsentropicPressureGradient(
		grid, 4, 'periodic', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	s   = state['air_isentropic_density'].values
	mtg = state['montgomery_potential'].values

	_s2   = np.concatenate((s[-2:-1, :, :], s, s[1:2, :, :]), axis=0)
	s2    = np.concatenate((_s2[:, -2:-1, :], _s2, _s2[:, 1:2, :]), axis=1)
	_s4   = np.concatenate((s[-3:-1, :, :], s, s[1:3, :, :]), axis=0)
	s4    = np.concatenate((_s4[:, -3:-1, :], _s4, _s4[:, 1:3, :]), axis=1)
	_mtg2 = np.concatenate((mtg[-2:-1, :, :], mtg, mtg[1:2, :, :]), axis=0)
	mtg2  = np.concatenate((_mtg2[:, -2:-1, :], _mtg2, _mtg2[:, 1:2, :]), axis=1)
	_mtg4 = np.concatenate((mtg[-3:-1, :, :], mtg, mtg[1:3, :, :]), axis=0)
	mtg4  = np.concatenate((_mtg4[:, -3:-1, :], _mtg4, _mtg4[:, 1:3, :]), axis=1)

	su_tnd2 = - s2[1:-1, 1:-1, :] * (mtg2[2:, 1:-1, :] - mtg2[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(su_tnd2, tendencies2['x_momentum_isentropic'])

	sv_tnd2 = - s2[1:-1, 1:-1, :] * (mtg2[1:-1, 2:, :] - mtg2[1:-1, :-2, :]) / (2. * dy)
	assert 'y_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(sv_tnd2, tendencies2['y_momentum_isentropic'])

	su_tnd4 = - s4[2:-2, 2:-2, :] * \
			  (mtg4[:-4, 2:-2, :] - 8. * mtg4[1:-3, 2:-2, :] +
			   8. * mtg4[3:-1, 2:-2, :] - mtg4[4:, 2:-2, :]) / (12. * dx)
	assert 'x_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(su_tnd4, tendencies4['x_momentum_isentropic'])

	sv_tnd4 = - s4[2:-2, 2:-2, :] * \
			  (mtg4[2:-2, :-4, :] - 8. * mtg4[2:-2, 1:-3, :] +
			   8. * mtg4[2:-2, 3:-1, :] - mtg4[2:-2, 4:, :]) / (12. * dy)
	assert 'y_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(sv_tnd4, tendencies4['y_momentum_isentropic'])


if __name__ == '__main__':
	pytest.main([__file__])
