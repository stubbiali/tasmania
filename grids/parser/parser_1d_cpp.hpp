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
