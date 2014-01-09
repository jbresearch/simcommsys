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

#include "stream_lut.h"
#include <cstring>

namespace libcomm {

// creation/destruction functions

template <class real>
stream_lut<real>::stream_lut(const char *filename, FILE *file, const int tau,
      const int m)
   {
   stream_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s == NULL) ? filename : s + 1;
   this->lutname = p;

   this->lut.init(tau);

   char buf[256];
   for (int i = 0; i < tau - m; i++)
      {
      do
         {
         assertalways(fscanf(file, "%[^\n]\n", buf) == 1);
         } while (buf[0] == '#');
      int y;
      sscanf(buf, "%d", &y);
      this->lut(i) = y;
      }
   for (int t = tau - m; t < tau; t++)
      this->lut(t) = fsm::tail;
   }

// Explicit instantiations

template class stream_lut<float> ;
template class stream_lut<double> ;
template class stream_lut<libbase::logrealfast> ;
} // end namespace
