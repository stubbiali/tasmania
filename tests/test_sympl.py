import numpy as np
import pytest


def test_import():
	try:
		import sympl
	except ImportError:
		print('Hint: did you install sympl?')

	import sys
	assert 'sympl' in sys.modules


def test_to_units():
	import sympl
	from tasmania.namelist import datatype

	domain_x, nx, dims_x, units_x = [-50, 50], 101, 'x', 'km'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=datatype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=datatype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	x_to_units = x.to_units('m')

	assert x_to_units[0] == -50.e3
	assert x_to_units[-1] == 50.e3


if __name__ == '__main__':
	pytest.main([__file__])
