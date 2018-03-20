import os
import numpy as np
import gridtools as gt


def definitions_average(in_field):
	"""
	A simple three-dimensional average operator.

	:param in_field	Input three-dimensional field

	:return Averaged field
	"""
	# Indeces
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	# Output
	out_field = gt.Equation()

	# Computations
	out_field[i, j, k] = (in_field[i, j, k] + in_field[i-1, j, k] + in_field[i+1, j, k] + \
						in_field[i, j-1, k] + in_field[i, j+1, k] + \
						in_field[i, j, k-1] + in_field[i, j, k+1]) / 7.

	return out_field


def function_average(in_field):
	out = np.zeros(in_field.shape, float)
	out[1:-1, 1:-1, 1:-1] = (in_field[1:-1, 1:-1, 1:-1] + in_field[:-2, 1:-1, 1:-1] + in_field[2:, 1:-1, 1:-1] + \
							in_field[1:-1, :-2, 1:-1] + in_field[1:-1, 2:, 1:-1] + \
							in_field[1:-1, 1:-1, :-2] + in_field[1:-1, 1:-1, 2:]) / 7.
	return out


# Initialize input and output
in_field = np.random.rand(50, 50, 10)
out_field = np.zeros((50, 50, 10), float) 

# Initialize and run the stencil
stencil = gt.NGStencil(definitions_func = definitions_average,
					inputs = {"in_field": in_field},
					outputs = {"out_field": out_field},
					domain = gt.domain.Rectangle((1,1,1), (48,48,8)),
					mode = gt.mode.NUMPY,
					vertical_direction = gt.vertical_direction.PARALLEL)
stencil.compute()

# Results validation
out = function_average(in_field)
print("Results validated : " + str(np.array_equal(out, out_field)))
