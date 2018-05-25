import gridtools as gt
import numpy as np
import pytest

def _defs_stencil_one_stage(in_a, in_b):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	out_c = gt.Equation()

	out_c[i, j, k] = (in_a[k-1] + in_a[k+1]) * \
					 (in_b[i+1, j] + in_b[i-1, j] + in_b[i, j+1] + in_b[i, j-1])

	return out_c

def _defs_stencil_two_stages(in_a, in_b):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	tmp = gt.Equation()
	out_c = gt.Equation()

	tmp[k] = in_a[k-1] + in_a[k+1]
	out_c[i, j, k] = tmp[k] * (in_b[i+1, j] + in_b[i-1, j] + in_b[i, j+1] + in_b[i, j-1])

	return out_c

def _defs_stencil_three_stages(in_a, in_b):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	tmp1 = gt.Equation()
	tmp2 = gt.Equation()
	out_c = gt.Equation()

	tmp1[i, j] = in_b[i+1, j] + in_b[i-1, j] + in_b[i, j+1] + in_b[i, j-1]
	tmp2[k] = in_a[k-1] + in_a[k+1]
	out_c[k] = tmp2[k] * tmp1[k]

	return out_c

def _stencil_numpy(in_a, in_b, out_c):
	out_c[1:-1, 1:-1, 1:-1] = (in_a[1:-1, 1:-1, :-2] + in_a[1:-1, 1:-1, 2:]) * \
							  (in_b[:-2, 1:-1, 1:-1] + in_b[2:, 1:-1, 1:-1] +
							   in_b[1:-1, :-2, 1:-1] + in_b[1:-1, 2:, 1:-1])

def _check_transform_indices(expression, indices):
	indices_transformations = [edge.get_indices_transformation() for edge in expression.get_edges()]
	for indices_transformation in indices_transformations:
		transformed_indices = indices_transformation.transform_indices(indices)
		assert(len(transformed_indices) == 3)
	for edge in expression.get_edges():
		_check_transform_indices(edge.get_expression_target(), indices)

def _check_source_indices(expression):
	indices_transformations = [edge.get_indices_transformation() for edge in expression.get_edges()]
	for indices_transformation in indices_transformations:
		assert(len(indices_transformation.get_source_indices()) == 3)
	for edge in expression.get_edges():
		_check_source_indices(edge.get_expression_target())

def test_transform_indices_one_stage():
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	in_a = gt.Equation()
	in_b = gt.Equation()
	out_c = gt.Equation()

	out_c[i, j, k] = (in_a[k-1] + in_a[k+1]) * \
					 (in_b[i+1, j] + in_b[i-1, j] + in_b[i, j+1] + in_b[i, j-1])
	
	edges_without_transformation = [edge for edge in out_c.expression.get_edges() if edge.get_indices_transformation() is None]
	assert(len(edges_without_transformation) == 0)

	_check_transform_indices(out_c.expression, (i, j, k))

def test_transform_indices_two_stages():
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	in_a = gt.Equation()
	in_b = gt.Equation()
	tmp_a = gt.Equation()
	out_c = gt.Equation()

	tmp_a[k] = in_a[k-1] + in_a[k+1]
	out_c[i, j, k] = tmp_a[k] * (in_b[i+1, j] + in_b[i-1, j] + in_b[i, j+1] + in_b[i, j-1])

	edges_without_transformation = [edge for edge in out_c.expression.get_edges() if edge.get_indices_transformation() is None]
	assert(len(edges_without_transformation) == 0)

	_check_transform_indices(out_c.expression, (i, j, k))

def test_source_indices_two_stages():
	in_a = gt.Equation()
	in_b = gt.Equation()

	out_c = _defs_stencil_two_stages(in_a, in_b)

	edges_without_transformation = [edge for edge in out_c.expression.get_edges() if edge.get_indices_transformation() is None]
	assert(len(edges_without_transformation) == 0)

	_check_source_indices(out_c.expression)

def test_source_indices_three_stages():
	in_a = gt.Equation()
	in_b = gt.Equation()

	out_c = _defs_stencil_three_stages(in_a, in_b)

	edges_without_transformation = [edge for edge in out_c.expression.get_edges() if edge.get_indices_transformation() is None]
	assert(len(edges_without_transformation) == 0)

	_check_source_indices(out_c.expression)

def test_compute_offsets_one_stage():
	nx = ny = 100
	nz = 50

	in_a = np.random.rand(nx, ny, nz)
	in_b = np.random.rand(nx, ny, nz)
	out_c = np.zeros_like(in_a)

	stencil_gt = gt.NGStencil(
		definitions_func = _defs_stencil_one_stage,
		inputs = {'in_a': in_a, 'in_b': in_b},
		outputs = {'out_c': out_c},
		domain = gt.domain.Rectangle((1, 1, 1), (nx-2, ny-2, nz-2)),
		mode = gt.mode.NUMPY
	)

	stencil_gt.compute()

	test_c = np.zeros_like(in_a)
	_stencil_numpy(in_a, in_b, test_c)

	assert(np.allclose(out_c, test_c))

def test_compute_offsets_two_stages():
	nx = ny = 100
	nz = 50

	in_a = np.random.rand(nx, ny, nz)
	in_b = np.random.rand(nx, ny, nz)
	out_c = np.zeros_like(in_a)

	stencil_gt = gt.NGStencil(
		definitions_func = _defs_stencil_two_stages,
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
