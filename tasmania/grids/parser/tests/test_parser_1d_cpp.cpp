#include "parser_1d_cpp.hpp"

int main()
{
	// Expression and evaluation points
	string expr_string("exp(x) + 1");
	vector<double> eval_points({1., 0.1, 3.});

	// Parse
	parser_1d_cpp parser(expr_string, eval_points);
	auto values = parser.parse();

	// Output
	cout << "f(x) = " << expr_string << endl;
	for (int i = 0; i < values.size(); ++i)
		cout << "f(" << eval_points[i]
			 << ") = " << values[i] << endl;
}
