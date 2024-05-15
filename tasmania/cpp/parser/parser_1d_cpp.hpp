/*
 * Tasmania
 *
 * Copyright (c) 2018-2024, ETH Zurich
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
/*! \file parser_1d_cpp.hpp */

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

/*! Parser for a one-dimensional expression in the independent variable \f$ x \f$. */
class parser_1d_cpp
{
	private:
		/*! The expression. */
		string expression_string;

		/*! The evaluation points. */
		vector<double> eval_points;

	public:
		/*! Constructor.
			\param expr	The expression to parse. It must satisfy two requirements:
						- the independent variable is \f$ x \f$;
						- fully C++-compliant.
			\param x	Evaluation points, i.e., where the expression should be evaluated. */
		parser_1d_cpp(const char * expr, const vector<double> & x);

		/*! Parse and evaluate the expression.
			\return		Expression value at the evaluation points. */
		vector<double> evaluate() const;
};
