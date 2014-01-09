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

#ifndef __fastsecant_h
#define __fastsecant_h

#include "config.h"
#include "secant.h"
#include "vector.h"

namespace libbase {

/*!
 * \brief   Semi-cached root-finding by Secant method.
 * \author  Johann Briffa
 *
 * Speeded-up version of the secant method module - we build a cache on seeding
 * which we then use to initialise the starting points for the algorithm.
 */

class fastsecant : public secant {
   vector<double> m_vdCache;
   double m_dMin, m_dMax, m_dStep;
public:
   fastsecant(double(*func)(double) = NULL);
   void init(const double x1, const double x2, const int n);
   double solve(const double y);
   double operator()(const double y)
      {
      return solve(y);
      }
};

} // end namespace

#endif

