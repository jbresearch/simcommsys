/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __secant_h
#define __secant_h

#include "config.h"
#include <iostream>

namespace libbase {

/*!
 * \brief   Root-finding by Secant method.
 * \author  Johann Briffa
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

