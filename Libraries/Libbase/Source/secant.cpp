/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "secant.h"

#include <algorithm>
#include <stdlib.h>
#include <math.h>

namespace libbase {

// exported functions

secant::secant(double (*func)(double))
   {
   bind(func);
   init(0,1);
   accuracy(1e-10);
   maxiter(1000);
   }

void secant::init(const double x1, const double x2)
   {
   init_x1 = x1;
   init_x2 = x2;
   }

double secant::solve(const double y)
   {
   assertalways(f != NULL);

   // Initialise
   double x1 = init_x1;
   double x2 = init_x2;
   double y1 = (*f)(x1) - y;
   double y2 = (*f)(x2) - y;

   if(fabs(y2) < fabs(y1))
      {
      std::swap(x1, x2);
      std::swap(y1, y2);
      }

   for(int i=0; i<max_iter; i++)
      {
      double dx = (x2-x1)*y1/(y1-y2);
      x2 = x1;
      y2 = y1;
      x1 += dx;
      y1 = (*f)(x1) - y;
      if(y1 == 0.0 || fabs(dx) < min_dx)
         return x1;
      }

   std::cerr << "FATAL ERROR (secant): Maximum number of iterations exceeded.\n";
   exit(1);
   }

}; // end namespace
