import gridtools as gt
import numpy as np
import os
import pytest

def _defs_stencil(src):
	k = gt.Index(axis = 2)

	dst = gt.Equation()

	dst[k] = src[k-1] + src[k-2]

	return dst

def _fibonacci(n):
	old = 1
	now = 1
	for i in range(2,n):
		now += old
		old = now - old
	return now

def test_fibonacci():
	nx = ny = 100
	nz = 50 
	src = np.ones((nx, ny, nz), float)
	dst = src

	stencil = gt.NGStencil(definitions_func = _defs_stencil,
						inputs = {"src": src},
						outputs = {"dst": dst},
						domain = gt.domain.Rectangle((0, 0, 2), (nx-1, ny-1, nz-1)),
						mode = gt.mode.NUMPY,
						vertical_direction = gt.vertical_direction.FORWARD)
	stencil.compute()

	assert(dst[0,0,nz-1] == _fibonacci(nz))

if __name__ == '__main__':
	pytest.main([__file__])
