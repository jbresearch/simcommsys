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

#include "vale96int.h"

namespace libcomm {

// initialisation functions

template <class real>
vale96int<real>::vale96int()
   {
   // set name and forced tail length
   this->lutname = "vale96int";
   this->m = 0;
   // build LUT
   const int tau = 34;
   const int a[] = {16, 29, 9, 10, 14, 6, 31, 8, 12, 22, 17, 33, 34, 23, 24,
         19, 32, 30, 13, 2, 21, 25, 26, 3, 28, 20, 27, 7, 5, 15, 4, 11, 18, 1};
   this->lut.init(tau);
   this->lut = -1;
   for (int i = 0; i < tau; i++)
      {
      const int ndx = a[i] - 1;
      // check for duplicate entries
      assertalways(this->lut(ndx) == -1);
      this->lut(ndx) = i;
      }
   }

// Explicit instantiations

template class vale96int<float> ;
template class vale96int<double> ;
template class vale96int<libbase::logrealfast> ;
} // end namespace
