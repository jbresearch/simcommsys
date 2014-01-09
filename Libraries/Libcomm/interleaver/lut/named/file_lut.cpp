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

#include "file_lut.h"
#include <cstdio>
#include <cstring>

namespace libcomm {

// creation/destruction functions

template <class real>
file_lut<real>::file_lut(const char *filename, const int tau, const int m)
   {
   file_lut::m = m;

   const char *s = strrchr(filename, libbase::DIR_SEPARATOR);
   const char *p = (s == NULL) ? filename : s + 1;
   this->lutname = p;

   this->lut.init(tau);

   char buf[256];
   FILE *file = fopen(filename, "rb");
   if (file == NULL)
      {
      std::cerr << "FATAL ERROR (file_lut): Cannot open LUT file (" << filename
            << ")." << std::endl;
      exit(1);
      }
   for (int i = 0; i < tau - m; i++)
      {
      do
         {
         assertalways(fscanf(file, "%[^\n]\n", buf) == 1);
         } while (buf[0] == '#');
      int x, y;
      sscanf(buf, "%d%d", &x, &y);
      if (x != i)
         {
         std::cerr << "FATAL ERROR (file_lut): unexpected entry for line " << i
               << ": " << x << ", " << y << std::endl;
         exit(1);
         }
      this->lut(i) = y;
      }
   for (int t = tau - m; t < tau; t++)
      this->lut(t) = fsm::tail;
   fclose(file);
   }

// Explicit instantiations

template class file_lut<float> ;
template class file_lut<double> ;
template class file_lut<libbase::logrealfast> ;

} // end namespace
