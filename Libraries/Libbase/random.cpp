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

#include "random.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Track reseeding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

void random::seed(int32u s)
   {
#if DEBUG>=2
   std::cerr << "DEBUG: random (" << this << ") reseeded with " << s
         << " after " << counter << " steps." << std::endl;
#endif
#ifndef NDEBUG
   counter = 0;
   initialized = true;
#endif
   // this makes sure any stored gval is discarded
   next_gval_available = false;
   // initialize underlying generator
   init(s);
   }

double random::gval()
   {
   if (next_gval_available)
      {
      next_gval_available = false;
      return next_gval;
      }

   double v1, v2, rsq;
   do
      {
      v1 = 2.0 * fval_closed() - 1.0;
      v2 = 2.0 * fval_closed() - 1.0;
      rsq = (v1 * v1) + (v2 * v2);
      } while (rsq >= 1.0 || rsq == 0.0);
   double fac = sqrt(-2.0 * log(rsq) / rsq);
   next_gval = v2 * fac;
   next_gval_available = true;
   return (v1 * fac);
   }

} // end namespace
