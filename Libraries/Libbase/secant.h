#ifndef __secant_h
#define __secant_h

#include "config.h"
#include <iostream>

namespace libbase {

/*!
 * \brief   Root-finding by Secant method.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

class secant {
   double (*f)(double);
   double init_x1, init_x2, min_dx;
   int max_iter;
public:
   explicit secant(double(*func)(double)=NULL);
   void bind(double(*func)(double))
      {
      f = func;
      }
   //! Set function domain to be explored
   void init(const double x1, const double x2);
   //! Set resolution of result
   void accuracy(const double dx)
      {
      min_dx = dx;
      }
   //! Set maximum number of iterations for secant method
   void maxiter(const int n)
      {
      assert(n >= 1);
      max_iter = n;
      }
   //! Find input value for which function value is y
   double solve(const double y);
   //! Function notation for solve()
   double operator()(const double y)
      {
      return solve(y);
      }
};

} // end namespace

#endif

