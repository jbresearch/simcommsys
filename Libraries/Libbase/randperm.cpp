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

#include "randperm.h"

namespace libbase {

void randperm::init(const int N, random& r)
   {
   assert(N >= 0);
   // initialize array to hold permuted positions
   lut.init(N);
   if (N == 0)
      return;
   lut = -1;
   // create the permutation vector
   for (int i = 0; i < N; i++)
      {
      int j;
      do
         {
         j = r.ival(N);
         } while (lut(j) >= 0);
      lut(j) = i;
      }
   }

} // end namespace
