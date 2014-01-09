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

#include "randgen.h"

namespace libbase {

const int32s randgen::mbig = 1000000000L;
const int32s randgen::mseed = 161803398L;

void randgen::init(int32u s)
   {
   next = 0L;
   nextp = 31L;
   mj = (mseed - s) % mbig;
   ma[55] = mj;
   int32s mk = 1;
   for (int i = 1; i <= 54; i++)
      {
      int ii = (21 * i) % 55;
      ma[ii] = mk;
      mk = mj - mk;
      if (mk < 0)
         mk += mbig;
      mj = ma[ii];
      }
   for (int k = 1; k <= 4; k++)
      for (int i = 1; i <= 54; i++)
         {
         ma[i] -= ma[1 + (i + 30) % 55];
         if (ma[i] < 0)
            ma[i] += mbig;
         }
   }

void randgen::advance()
   {
   if (++next >= 56)
      next = 1;
   if (++nextp >= 56)
      nextp = 1;
   mj = ma[next] - ma[nextp];
   if (mj < 0)
      mj += mbig;
   ma[next] = mj;
   }

} // end namespace
