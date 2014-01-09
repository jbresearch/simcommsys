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

#ifndef HAMMING_H_
#define HAMMING_H_

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Compute Hamming Distance
 * \author  Johann Briffa
 *
 * A method that computes the Hamming distance between two sequences.
 * Templatized for any type for which a definition of equality exists.
 */

template <class T>
int hamming(const vector<T>& s, const vector<T>& t)
   {
   const int m = s.size();
   const int n = t.size();

   assertalways(m == n);

   // initialize distance
   int d = 0;

   // fill in the rest of the table
   for (int i = 0; i < n; i++)
      {
      if (s(i) == t(i))
         continue;
      else
         d++;
      }

   return d;
   }

} // end namespace

#endif /* HAMMING_H_ */
