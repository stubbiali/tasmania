/*! \file parser_2d_cpp.hpp */

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

/*! Parser for a two-dimensional expression in the independent variables
  	\f$ x \f$ and \f$ y \f$. The expression is evaluated on a rectangular 
	grid \f$ (x_i, y_j)_{1 \leq i \leq N_x, \, 1 \leq j \leq N_y} \f$. */
class parser_2d_cpp
{
	private:
		/*! The expression. */
		string expression_string;

		/*! The \f$ x \f$-coordinates of the evaluation points. */
		vector<double> eval_points_x;

		/*! The \f$ y \f$-coordinates of the evaluation points. */
		vector<double> eval_points_y;

	public:
		/*! Constructor. 
			\param expr	The expression to parse. It must satisfy two requirements:
						- the independent variableis are \f$ x \f$ and \f$ y \f$;
						- fully C++-compliant.
			\param x	\f$ x \f$-coordinates of the evaluation points, i.e., the 
						points where the expression should be evaluated.
			\param y	\f$ y \f$-coordinates of the evaluation points, i.e., the 
						points where the expression should be evaluated. */
		parser_2d_cpp(const char * expr, const vector<double> & x, const vector<double> & y);

		/*! Parse and evaluate the expression. 
			\return		Expression evaluations, arranged in a 2d-array with
						size \f$ (N_x, \, N_y) \f$. */
		vector< vector<double> > evaluate() const;
};
