import os
import numpy as np
import gridtools as gt


def definitions_fibonacci(src):
	# Indices
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	# Output
	dst = gt.Equation()

	# Computations
	dst[i, j, k] = src[i, j, k-1] + src[i, j, k-2]

	return dst


n = 50 
src = np.ones((1, 1, n), float)
dst = src

stencil = gt.NGStencil(definitions_func = definitions_fibonacci,
					inputs = {"src": src},
					outputs = {"dst": dst},
					domain = gt.domain.Rectangle((0,0,2), (0,0,n-1)),
					mode = gt.mode.NUMPY,
					vertical_direction = gt.vertical_direction.FORWARD)
stencil.compute()

print("{}th Fibonacci number: {}".format(n, dst[0,0,n-1]))	
