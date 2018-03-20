from gridtools.frontend.crappy import index as idx
from gridtools.frontend.crappy import expression as expr


def test_default_expression():
    A = expr.Equation()

    assert type(A.expression) == expr.ExpressionNamed
    assert A.expression._name_given_by_user == "A"
    assert len(A.expression.get_subexpressions()) == 0


def test_assignment():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    B[i, j] = A[i, j]

    assert type(B.expression) == expr.ExpressionNamed
    assert B.expression._name_given_by_user == "B"
    assert len(B.expression.get_subexpressions()) == 1

    assert type(B.expression.get_subexpressions()[0]) == expr.ExpressionNamed
    assert B.expression.get_subexpressions()[0] == A.expression

    assert A.expression._name_given_by_user == "A"
    assert len(A.expression.get_subexpressions()) == 0


def test_assignment_scalar():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()

    A[i, j] = 10

    assert type(A.expression) == expr.ExpressionNamed
    assert A.expression._name_given_by_user == "A"
    assert len(A.expression.get_subexpressions()) == 1

    scalar = A.expression.get_subexpressions()[0]
    assert type(scalar) == expr.ExpressionConstant
    assert scalar._value == 10
    assert len(scalar.get_subexpressions()) == 0


def test_two_nested_binary_operators():
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()
    C = expr.Equation()

    B[i, j] = A[i, j] + 11
    C[i, j] = B[i, j] + 10

    assert type(C.expression) == expr.ExpressionNamed
    assert C.expression._name_given_by_user == "C"
    assert len(C.expression.get_subexpressions()) == 1

    first_plus = C.expression.get_subexpressions()[0]
    assert type(first_plus) == expr.ExpressionBinaryOperator
    assert first_plus._operator == "+"
    assert len(first_plus.get_subexpressions()) == 2

    first_rhs = first_plus.get_expression_rhs()
    assert type(first_rhs) == expr.ExpressionConstant
    assert first_rhs._value == 10
    assert len(first_rhs.get_subexpressions()) == 0

    first_lhs = first_plus.get_expression_lhs()
    assert type(first_lhs) == expr.ExpressionNamed
    assert first_lhs == B.expression
    assert first_lhs._name_given_by_user == "B"
    assert len(first_lhs.get_subexpressions()) == 1

    second_plus = first_lhs.get_subexpressions()[0]
    assert type(second_plus) == expr.ExpressionBinaryOperator
    assert second_plus._operator == "+"
    assert len(second_plus.get_subexpressions()) == 2

    second_rhs = second_plus.get_expression_rhs()
    assert type(second_rhs) == expr.ExpressionConstant
    assert second_rhs._value == 11
    assert len(second_rhs.get_subexpressions()) == 0

    second_lhs = second_plus.get_expression_lhs()
    assert type(second_lhs) == expr.ExpressionNamed
    assert second_lhs == A.expression
    assert second_lhs._name_given_by_user == "A"
    assert len(second_lhs.get_subexpressions()) == 0


def test_binary_operators():
    _test_binary_operator("+")
    _test_binary_operator("-")
    _test_binary_operator("*")
    _test_binary_operator("/")


def _test_binary_operator(operator):

    # test with scalar on the operator's right hand side
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    statement = "A[i, j] {} 10".format(operator)
    B[i, j] = eval(statement)

    assert type(B.expression) == expr.ExpressionNamed
    assert B.expression._name_given_by_user == "B"
    assert len(B.expression.get_subexpressions()) == 1

    plus = B.expression.get_subexpressions()[0]
    assert type(plus) == expr.ExpressionBinaryOperator
    assert plus._operator == operator
    assert len(plus.get_subexpressions()) == 2

    rhs = plus.get_expression_rhs()
    assert type(rhs) == expr.ExpressionConstant
    assert rhs._value == 10
    assert len(rhs.get_subexpressions()) == 0

    lhs = plus.get_expression_lhs()
    assert type(lhs) == expr.ExpressionNamed
    assert lhs == A.expression
    assert lhs._name_given_by_user == "A"
    assert len(lhs.get_subexpressions()) == 0

    # test with scalar on the operator's left hand side
    i = idx.Index()
    j = idx.Index()
    A = expr.Equation()
    B = expr.Equation()

    statement = "10 {} A[i, j]".format(operator)
    B[i, j] = eval(statement)

    assert type(B.expression) == expr.ExpressionNamed
    assert B.expression._name_given_by_user == "B"
    assert len(B.expression.get_subexpressions()) == 1

    plus = B.expression.get_subexpressions()[0]
    assert type(plus) == expr.ExpressionBinaryOperator
    assert plus._operator == operator
    assert len(plus.get_subexpressions()) == 2

    lhs = plus.get_expression_lhs()
    assert type(lhs) == expr.ExpressionConstant
    assert lhs._value == 10
    assert len(lhs.get_subexpressions()) == 0

    rhs = plus.get_expression_rhs()
    assert type(rhs) == expr.ExpressionNamed
    assert rhs == A.expression
    assert rhs._name_given_by_user == "A"
    assert len(rhs.get_subexpressions()) == 0
