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

#ifndef __mpreal_h
#define __mpreal_h

#include "config.h"
#include <iostream>

namespace libbase {

/*!
 * \brief   Multi-Precision Arithmetic.
 * \author  Johann Briffa
 *
 * \version 1.01 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.02 (15 Jun 2002)
 * - changed all pow() functions to more accurately specify that we want to
 * call double pow(double, int)
 * - changed 'flags' variable in implementation file from type int to type
 * ios::fmtflags, as it's supposed to be.
 *
 * \version 1.10 (26 Oct 2006)
 * - defined class and associated data within "libbase" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 *
 * \version 1.11 (17 Jul 2007)
 * - changed references to isinf() and isnan() back to global namespace, in accord with
 * config.h 3.23.
 */

class mpreal {
   static const double base;
   int32s exponent;
   long double mantissa;
   void normalise();
public:
   mpreal(const double m = 0);
   operator double() const;

   mpreal& operator-();
   mpreal& operator+=(const mpreal& a);
   mpreal& operator-=(const mpreal& a);
   mpreal& operator*=(const mpreal& a);
   mpreal& operator/=(const mpreal& a);

   friend std::ostream& operator<<(std::ostream& s, const mpreal& x);
};

// Derived Operations

inline mpreal operator+(const mpreal& a, const mpreal& b)
   {
   mpreal result = a;
   result += b;
   return result;
   }

inline mpreal operator-(const mpreal& a, const mpreal& b)
   {
   mpreal result = a;
   result -= b;
   return result;
   }

inline mpreal operator*(const mpreal& a, const mpreal& b)
   {
   mpreal result = a;
   result *= b;
   return result;
   }

inline mpreal operator/(const mpreal& a, const mpreal& b)
   {
   mpreal result = a;
   result /= b;
   return result;
   }

} // end namespace

#endif
