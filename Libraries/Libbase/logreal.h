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

#ifndef __logreal_h
#define __logreal_h

#include "config.h"
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>

namespace libbase {

/*!
 * \brief   Logarithm Arithmetic.
 * \author  Johann Briffa
 */

class logreal {
   double logval;
   logreal& operator-();
   logreal& operator-=(const logreal& a);
public:
   logreal(const double m = 0);
   operator double() const;

   logreal& operator+=(const logreal& a);
   logreal& operator*=(const logreal& a);
   logreal& operator/=(const logreal& a);

   friend std::ostream& operator<<(std::ostream& s, const logreal& x);
};

inline logreal::logreal(const double m)
   {
   if (m < 0)
      {
      std::cerr << "FATAL ERROR (logreal): Negative numbers cannot be used." << std::endl;
      exit(1);
      }
   logval = (m == 0) ? DBL_MAX : -log(m);
   }

inline logreal::operator double() const
   {
   return exp(-logval);
   }

inline logreal& logreal::operator+=(const logreal& a)
   {
   if (logval < a.logval)
      logval = logval - log(1 + exp(logval - a.logval));
   else
      logval = a.logval - log(1 + exp(a.logval - logval));
   return *this;
   }

inline logreal& logreal::operator*=(const logreal& a)
   {
   logval += a.logval;
   if (isinf(logval))
      logval = isinf(logval) * DBL_MAX;
   return *this;
   }

inline logreal& logreal::operator/=(const logreal& a)
   {
   logval -= a.logval;
   if (isinf(logval))
      logval = isinf(logval) * DBL_MAX;
   return *this;
   }

// The following functions operate through the above - no need to make them friends

inline logreal operator+(const logreal& a, const logreal& b)
   {
   logreal result = a;
   result += b;
   return result;
   }

inline logreal operator*(const logreal& a, const logreal& b)
   {
   logreal result = a;
   result *= b;
   return result;
   }

inline logreal operator/(const logreal& a, const logreal& b)
   {
   logreal result = a;
   result /= b;
   return result;
   }

} // end namespace

#endif
