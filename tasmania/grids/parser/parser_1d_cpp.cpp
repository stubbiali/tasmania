/*
 * Tasmania
 *
 * Copyright (c) 2018-2019, ETH Zurich
 * All rights reserved.
 *
 * This file is part of the Tasmania project. Tasmania is free software:
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
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



