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

#include "sparse.h"

namespace libbase {

/*!
 * \brief Set up LUT with the lowest weight codewords
 */
int sparse::fill(int i, bitfield suffix, int weight)
   {
   // stop here if we've reached the end
   if (i >= lut.size())
      return i;
   // otherwise, it all depends on the weight we're considering
   bitfield b;
#ifndef NDEBUG
   if (lut.size() > 2)
      trace << "Starting fill with:\t" << suffix << "\t" << weight << std::endl;
#endif
   if (weight == 0)
      lut(i++) = suffix;
   else
      {
      weight--;
      if (suffix.size() == 0)
         i = fill(i, suffix, weight);
      for (b = bitfield("1"); b.size() + suffix.size() + weight <= n; b = b
            + bitfield("0"))
         i = fill(i, b + suffix, weight);
      }
   return i;
   }

void sparse::init(const int q, const int n)
   {
   assert(q >= 0);
   assert(q <= (1 << n));
   // set codeword size
   this->n = n;
   // initialize array to hold permuted positions
   lut.init(q);
   if (q == 0)
      return;
   // set up codebook
   fill(0, bitfield(""), n);
   }

} // end namespace
