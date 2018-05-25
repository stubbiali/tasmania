import gridtools as gt
import numpy as np
import pytest

def _defs_stencil(in_a, in_b):
	k = gt.Index(axis = 2)
	i = gt.Index(axis = 0)
	j = gt.Index(axis = 1)

	out_c = gt.Equation()

	out_c[i, j, k] = (in_a[i, j, k-1] + in_a[i, j, k+1]) * \
					 (in_b[i+1, j, k] + in_b[i-1, j, k] +
					  in_b[i, j+1, k] + in_b[i, j-1, k])

	return out_c

def _stencil_numpy(in_a, in_b, out_c):
	out_c[1:-1, 1:-1, 1:-1] = (in_a[1:-1, 1:-1, :-2] + in_a[1:-1, 1:-1, 2:]) * \
							  (in_b[:-2, 1:-1, 1:-1] + in_b[2:, 1:-1, 1:-1] +
							   in_b[1:-1, :-2, 1:-1] + in_b[1:-1, 2:, 1:-1])

def test_index_axis():
	nx = ny = 100
	nz = 50

	in_a = np.random.rand(nx, ny, nz)
	in_b = np.random.rand(nx, ny, nz)
	out_c = np.zeros_like(in_a)

	stencil_gt = gt.NGStencil(
		definitions_func = _defs_stencil,
		inputs = {'in_a': in_a, 'in_b': in_b},
		outputs = {'out_c': out_c},
		domain = gt.domain.Rectangle((1, 1, 1), (nx-2, ny-2, nz-2)),
		mode = gt.mode.NUMPY
	)

	stencil_gt.compute()

	test_c = np.zeros_like(in_a)
	_stencil_numpy(in_a, in_b, test_c)

	assert(np.allclose(out_c, test_c))

if __name__ == '__main__':
	pytest.main([__file__])
