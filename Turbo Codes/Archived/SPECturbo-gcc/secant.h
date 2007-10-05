#ifndef __secant_h
#define __secant_h

#include "config.h"
#include "vcs.h"

#include <iostream.h>

extern const vcs secant_version;

class secant {
   double	(*f)(double);
   double	init_x1, init_x2, min_dx;
   int		max_iter;
public:
   secant(double (*func)(double) = NULL);
   void bind(double (*func)(double));
   void seed(double x1, double x2);
   void accuracy(double dx);
   void maxiter(int n);
   double solve(double y);
   double operator()(double y);
};

inline double secant::operator()(double y)
   {
   return solve(y);
   }
   
#endif

