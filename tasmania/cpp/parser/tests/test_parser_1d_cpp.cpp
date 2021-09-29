/*
 * Tasmania
 *
 * Copyright (c) 2018-2021, ETH Zurich
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
