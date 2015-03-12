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

#include "histogram.h"
#include <cfloat>

namespace libbase {

void histogram::initbins(const double min, const double max, const int n)
   {
   step = (max - min) / double(n);
   x.init(n);
   for (int i = 0; i < n; i++)
      x(i) = min + i * step;
   }

histogram::histogram(const vector<double>& a, const int n)
   {
   initbins(a.min(), a.max(), n);

   y.init(n);
   y = 0;
   for (int i = 0; i < a.size(); i++)
      {
      for (int k = n - 1; k >= 0; k--)
         if (a(i) >= x(k))
            {
            y(k)++;
            break;
            }
      }
   }

} // end namespace
