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

#include "secant.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace libbase {

// exported functions

secant::secant(double(*func)(double))
   {
   bind(func);
   init(0, 1);
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

   if (fabs(y2) < fabs(y1))
      {
      std::swap(x1, x2);
      std::swap(y1, y2);
      }

   for (int i = 0; i < max_iter; i++)
      {
      double dx = (x2 - x1) * y1 / (y1 - y2);
      x2 = x1;
      y2 = y1;
      x1 += dx;
      y1 = (*f)(x1) - y;
      if (y1 == 0.0 || fabs(dx) < min_dx)
         return x1;
      }

   std::cerr
         << "FATAL ERROR (secant): Maximum number of iterations exceeded." << std::endl;
   exit(1);
   }

} // end namespace
