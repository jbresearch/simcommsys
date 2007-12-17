#ifndef __secant_h
#define __secant_h

#include "config.h"
#include "vcs.h"
#include <iostream>

namespace libbase {

/*!
   \brief   Root-finding by Secant method.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.02 (11-12 Jun 2002)
  * added <algorithm> to supply the definition of the swap() function (which has
  been removed from config).
  * modified the definition of 'f' so that its parameter is non-const; this does not
  make any difference since the parameter is not passed by reference. It also allows
  us greater flexibility in using the class, and was in fact necessitated by the
  similar change in itfunc 1.04.

  Version 1.10 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

class secant {
   static const vcs version;
   double       (*f)(double);
   double       init_x1, init_x2, min_dx;
   int          max_iter;
public:
   secant(double (*func)(double) = NULL);
   void bind(double (*func)(double)) { f = func; };
   void seed(const double x1, const double x2);
   void accuracy(const double dx) { min_dx = dx; };
   void maxiter(const int n) { max_iter = n; };
   double solve(const double y);
   double operator()(const double y) { return solve(y); };
};

}; // end namespace

#endif

