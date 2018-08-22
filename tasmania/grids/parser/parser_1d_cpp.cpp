/*! \file parser_1d_cpp.cpp */

#include "exprtk.hpp"
#include "parser_1d_cpp.hpp"

// Shortcuts
typedef exprtk::symbol_table<double>	symbol_table_t;
typedef exprtk::expression<double>		expression_t;
typedef exprtk::parser<double>			parser_t;

parser_1d_cpp::parser_1d_cpp(const char * expr, const vector<double> & x) :
	expression_string(expr), eval_points(x)
{
}

vector<double> parser_1d_cpp::evaluate() const
{
	double x;

	symbol_table_t symbol_table;
	symbol_table.add_variable("x", x);
	symbol_table.add_constants();

	expression_t expression;
	expression.register_symbol_table(symbol_table);

	parser_t parser;
	parser.compile(this->expression_string, expression);

	vector<double> values(this->eval_points.size());
	for (size_t i = 0; i < values.size(); ++i)
	{
		x = this->eval_points[i];
		values[i] = expression.value();
	}

	return values;
}



