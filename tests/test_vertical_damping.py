from datetime import timedelta
import numpy as np
import pytest

from tasmania.dynamics.vertical_damping import VerticalDamping as VD


def test_rayleigh(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	dt = timedelta(seconds=27)

	ni, nj, nk = nx+1, ny, nz+1
	phi_now = np.random.rand(ni, nj, nk)
	phi_new = np.random.rand(ni, nj, nk)
	phi_ref = np.random.rand(ni, nj, nk)
	vd = VD.factory('rayleigh', (ni, nj, nk), grid, 15, 0.03, dtype=phi_now.dtype)
	phi_damp = vd(dt, phi_now, phi_new, phi_ref)

	assert phi_damp.shape == (ni, nj, nk)

	rmat, dd = vd._rmat, vd._damp_depth
	i, j, k, notk = slice(0, ni), slice(0, nj), slice(0, nk-dd), slice(nk-dd, nk)
	phi_damp_assert = np.zeros((ni, nj, nk), dtype=phi_now.dtype)
	phi_damp_assert[i, j, k] = phi_new[i, j, k] - dt.seconds * rmat[i, j, k] * \
							   (phi_now[i, j, k] - phi_ref[i, j, k])
	assert np.allclose(phi_damp_assert[i, j, k], phi_damp[i, j, k])
	assert np.allclose(phi_new[i, j, notk], phi_damp[i, j, notk])


if __name__ == '__main__':
	pytest.main([__file__])
