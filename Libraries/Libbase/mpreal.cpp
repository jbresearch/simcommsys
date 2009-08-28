/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "mpreal.h"
#include <math.h>
#include <stdlib.h>

namespace libbase {

const double mpreal::base = 10.0;

inline void mpreal::normalise()
   {
   if (mantissa == 0)
      {
      exponent = 0;
      return;
      }
   if (isinf(mantissa) || isnan(mantissa))
      return;
   int shift = (int) floor(log(fabs(mantissa)) / log(double(base)));
   mantissa *= pow(base, -shift);
   exponent += shift;
   }

// Conversion

mpreal::mpreal(const double m)
   {
   mantissa = m;
   exponent = 0;
   normalise();
   }

mpreal::operator double() const
   {
   return mantissa * pow(base, exponent);
   }

// Base Operations

mpreal& mpreal::operator-()
   {
   mantissa = -mantissa;
   return *this;
   }

mpreal& mpreal::operator+=(const mpreal& a)
   {
   if (mantissa == 0)
      {
      mantissa = a.mantissa;
      exponent = a.exponent;
      return *this;
      }
   if (a.mantissa == 0)
      return *this;

   if (exponent == a.exponent)
      mantissa += a.mantissa;
   else if (exponent > a.exponent)
      mantissa += a.mantissa * pow(base, a.exponent - exponent);
   else
      {
      mantissa *= pow(base, exponent - a.exponent);
      exponent = a.exponent;
      mantissa += a.mantissa;
      }
   normalise();
   return *this;
   }

mpreal& mpreal::operator-=(const mpreal& a)
   {
   mpreal x = a;
   *this += -x;
   return *this;
   }

mpreal& mpreal::operator*=(const mpreal& a)
   {
   mantissa *= a.mantissa;
   exponent += a.exponent;
   normalise();
   return *this;
   }

mpreal& mpreal::operator/=(const mpreal& a)
   {
   mantissa /= a.mantissa;
   exponent -= a.exponent;
   normalise();
   return *this;
   }

// Input/Output Operations

std::ostream& operator<<(std::ostream& s, const mpreal& x)
   {
   using std::ios;

   const ios::fmtflags flags = s.flags();
   s.setf(ios::fixed, ios::floatfield);
   s << x.mantissa;
   if (!(isinf(x.mantissa) || isnan(x.mantissa)))
      {
      s.setf(ios::showpos);
      s << "e" << x.exponent;
      }
   s.flags(flags);
   return s;
   }

} // end namespace
