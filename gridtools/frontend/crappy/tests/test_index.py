from gridtools.frontend.crappy import index as idx
from gridtools.frontend.crappy import expression as expr


def test_no_indices_transformation0():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()

    A[i, j] = 10

    trans = A.expression.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans, expected_rearrangments=[0, 1], expected_offsets=[0, 0])


def test_no_indices_transformation1():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[i, j]

    trans = B.expression.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans, expected_rearrangments=[0, 1], expected_offsets=[0, 0])


def test_no_indices_transformation2():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[i, j] + A[i, j]

    b = B.expression
    trans_b_to_plus = b.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_b_to_plus, expected_rearrangments=[0, 1], expected_offsets=[0, 0])

    plus = b.get_subexpressions()[0]
    trans_plus_to_lhs = plus.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_plus_to_lhs, expected_rearrangments=[0, 1], expected_offsets=[0, 0])

    trans_plus_to_rhs = plus.get_edges()[1].get_indices_transformation()
    _assert_transformation(trans_plus_to_rhs, expected_rearrangments=[0, 1], expected_offsets=[0, 0])


def test_indices_rearrangment():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[j, i]

    trans = B.expression.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans, expected_rearrangments=[1, 0], expected_offsets=[0, 0])


def test_indices_offset():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[i+10, j-11]

    trans = B.expression.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans, expected_rearrangments=[0, 1], expected_offsets=[10, -11])


def test_indices_rearrangement_and_offset():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[j+10, i-11]

    trans = B.expression.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans, expected_rearrangments=[1, 0], expected_offsets=[-11, 10])


def test_many_indices_transformations0():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[j+10, i-11] + A[i+20, j-21]

    b = B.expression
    trans_b_to_plus = b.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_b_to_plus, expected_rearrangments=[0, 1], expected_offsets=[0, 0])

    plus = b.get_subexpressions()[0]
    trans_plus_to_lhs = plus.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_plus_to_lhs, expected_rearrangments=[1, 0], expected_offsets=[-11, 10])

    trans_plus_to_rhs = plus.get_edges()[1].get_indices_transformation()
    _assert_transformation(trans_plus_to_rhs, expected_rearrangments=[0, 1], expected_offsets=[20, -21])


def test_many_indices_transformations1():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()
    C = expr.Equation()

    B[i, j] = A[j, i]
    C[i, j] = B[i+10, j-11]

    c = C.expression
    trans_c_to_b = c.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_c_to_b, expected_rearrangments=[0, 1], expected_offsets=[10, -11])

    b = c.get_subexpressions()[0]
    trans_b_to_a = b.get_edges()[0].get_indices_transformation()
    _assert_transformation(trans_b_to_a, expected_rearrangments=[1, 0], expected_offsets=[0, 0])


def _assert_transformation(trans, expected_rearrangments, expected_offsets):
    assert len(expected_rearrangments) == len(expected_offsets)
    for i in range(0, len(expected_rearrangments)):
        assert trans.get_rearranged_position_of_source_index(i) == expected_rearrangments[i]
        assert trans.get_offset_to_apply_to_source_index(i) == expected_offsets[i]
