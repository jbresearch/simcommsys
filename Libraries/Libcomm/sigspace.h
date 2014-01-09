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

#ifndef __sigspace_h
#define __sigspace_h

#include "config.h"
#include <cmath>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Signal Space point.
 * \author  Johann Briffa
 *
 * \version 1.01 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.02 (13 Mar 2002)
 * moved most functions to the implementation file instead of here.
 *
 * \version 1.10 (27 Mar 2002)
 * added two functions to make multiplication with double commutative. Also, made
 * passing of all double parameters direct, not by reference.
 *
 * \version 1.20 (30 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 *
 * \version 1.30 (14-15 Nov 2007)
 * - moved most functions here and made them inline.
 * - added equality and inequality operators.
 * - added unary '-' operator
 */

class sigspace {
   double inphase, quad;
public:
   // creator / destructor
   sigspace(const double i = 0, const double q = 0) :
         inphase(i), quad(q)
      {
      }

   double i() const
      {
      return inphase;
      }
   double q() const
      {
      return quad;
      }
   double r() const
      {
      return sqrt(i() * i() + q() * q());
      }
   double p() const
      {
      return atan2(q(), i());
      }
   operator double() const
      {
      return r();
      }

   // arithmetic operations
   sigspace& operator+=(const sigspace& a)
      {
      inphase += a.inphase;
      quad += a.quad;
      return *this;
      }
   sigspace& operator-=(const sigspace& a)
      {
      inphase -= a.inphase;
      quad -= a.quad;
      return *this;
      }
   sigspace& operator*=(const double a)
      {
      inphase *= a;
      quad *= a;
      return *this;
      }
   sigspace& operator/=(const double a)
      {
      inphase /= a;
      quad /= a;
      return *this;
      }

   // arithmetic operations - friends

   // comparison operations
   friend bool operator==(const sigspace& a, const sigspace& b)
      {
      return (a.inphase == b.inphase && a.quad == b.quad);
      }

   friend bool operator!=(const sigspace& a, const sigspace& b)
      {
      return (a.inphase != b.inphase || a.quad != b.quad);
      }

   // arithmetic operations - unary
   friend sigspace operator-(const sigspace& a)
      {
      return sigspace(-a.inphase, -a.quad);
      }

   // arithmetic operations - friends
   friend sigspace operator+(const sigspace& a, const sigspace& b)
      {
      sigspace c = a;
      c += b;
      return c;
      }
   friend sigspace operator-(const sigspace& a, const sigspace& b)
      {
      sigspace c = a;
      c -= b;
      return c;
      }
   friend sigspace operator*(const sigspace& a, const double b)
      {
      sigspace c = a;
      c *= b;
      return c;
      }
   friend sigspace operator/(const sigspace& a, const double b)
      {
      sigspace c = a;
      c /= b;
      return c;
      }
   friend sigspace operator*(const double a, const sigspace& b)
      {
      sigspace c = b;
      c *= a;
      return c;
      }
   friend sigspace operator/(const double a, const sigspace& b)
      {
      sigspace c = b;
      c /= a;
      return c;
      }

   // stream input / output
   friend std::ostream& operator<<(std::ostream& s, const sigspace& x);
   friend std::istream& operator>>(std::istream& s, sigspace& x);
};

} // end namespace

#endif
