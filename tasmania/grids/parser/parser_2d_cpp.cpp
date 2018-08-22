/*! \file parser_2d_cpp.cpp */

#include "exprtk.hpp"
#include "parser_2d_cpp.hpp"

// Shortcuts
typedef exprtk::symbol_table<double>	symbol_table_t;
typedef exprtk::expression<double>		expression_t;
typedef exprtk::parser<double>			parser_t;

parser_2d_cpp::parser_2d_cpp(const char * expr, const vector<double> & x, const vector<double> & y) :
	expression_string(expr), eval_points_x(x), eval_points_y(y)
{
}

vector< vector<double> > parser_2d_cpp::evaluate() const
{
	double x, y;

	symbol_table_t symbol_table;
	symbol_table.add_variable("x", x);
	symbol_table.add_variable("y", y);
	symbol_table.add_constants();

	expression_t expression;
	expression.register_symbol_table(symbol_table);

	parser_t parser;
	parser.compile(this->expression_string, expression);

	vector< vector<double> > values(this->eval_points_x.size());
	for (size_t i = 0; i < values.size(); ++i)
	{
		values[i].resize(this->eval_points_y.size());
		x = this->eval_points_x[i];
		for (size_t j = 0; j < values[i].size(); ++j)
		{
			y = this->eval_points_y[j];
			values[i][j] = expression.value();
		}
	}

	return values;
}



