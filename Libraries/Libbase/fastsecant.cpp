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

#include "fastsecant.h"
#include <cstdlib>
#include <cmath>

namespace libbase {

// exported functions

fastsecant::fastsecant(double(*func)(double)) :
   secant(func)
   {
   }

void fastsecant::init(const double x1, const double x2, const int n)
   {
   m_dMin = x1;
   m_dMax = x2;
   m_dStep = (x2 - x1) / double(n - 1);
   m_vdCache.init(n);
   double x = m_dMin;
   for (int i = 0; i < n; i++)
      {
      m_vdCache(i) = secant::solve(x);
      x += m_dStep;
      }
   }

double fastsecant::solve(const double y)
   {
   const int i = int(floor((y - m_dMin) / m_dStep));
   const int j = int(ceil((y - m_dMin) / m_dStep));
   if (i == j)
      return m_vdCache(i);
   else if (i >= 0 && j < m_vdCache.size())
      {
      const double x1 = m_vdCache(i);
      const double x2 = m_vdCache(j);
      secant::init(x1, x2);
      }
   else
      secant::init(-1, 1);

   return secant::solve(y);
   }

} // end namespace
