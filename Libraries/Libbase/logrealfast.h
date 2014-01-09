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

#ifndef __logrealfast_h
#define __logrealfast_h

#include "config.h"
#include "itfunc.h"
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of out-of-range values (infinity and zero)
// 3 - Log difference values and the errors for all LUT access to a file
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Fast Logarithm Arithmetic.
 * \author  Johann Briffa
 *
 * Implements log-scale arithmetic with table-lookup for speeding up
 * addition. The choice of LUT size and range is optimized at 128k entries
 * over [0,12].
 *
 * \note Constructor traps infinite values and NaN. Zero values are trapped
 * first; since zero is the default argument, there are many more calls
 * with this value than any other, so this should improve performance.
 *
 * \note Comparison operators are provided between variables of this kind -
 * these are required by the turbo decoder when taking a hard decision
 * (actually it only uses the greater-than operator, but all comparisons
 * are defined here). When these were not supplied, the comparison was
 * performed _after_ a conversion to double, which can easily cause
 * under- or over-flow, leading to a useless comparison.
 */

class logrealfast {
private:
   static const int lutsize;
   static const double lutrange;
   static double *lut_add;
   static double *lut_sub;
   static bool lutready;
#if DEBUG>=3
   static std::ofstream file;
#endif

private:
   double logval;
   void buildlut();
   static double convertfromdouble(const double m);
   static void ensurefinite(double& x);
   // define as private to ensure no-one uses it
   logrealfast& operator-();

public:
   // construction
   logrealfast()
      {
      if (!lutready)
         buildlut();
      }
   logrealfast(const double m)
      {
      if (!lutready)
         buildlut();
      logval = convertfromdouble(m);
      }
   logrealfast(const logrealfast& a)
      {
      // copy constructor need not check for lutready since at least one object
      // must have been created already.
      logval = a.logval;
      }
   // copy assignment
   logrealfast& operator=(const logrealfast& a)
      {
      logval = a.logval;
      return *this;
      }
   // conversion
   operator double() const
      {
      return exp(-logval);
      }
   logrealfast& operator=(const double m)
      {
      logval = convertfromdouble(m);
      return *this;
      }
   // arithmetic - unary
   logrealfast& operator+=(const logrealfast& b);
   logrealfast& operator-=(const logrealfast& b);
   logrealfast& operator*=(const logrealfast& a)
      {
      logval += a.logval;
      ensurefinite(logval);
      return *this;
      }
   logrealfast& operator/=(const logrealfast& a)
      {
      logval -= a.logval;
      ensurefinite(logval);
      return *this;
      }
   // comparison
   bool operator==(const logrealfast& a) const
      {
      return logval == a.logval;
      }
   bool operator!=(const logrealfast& a) const
      {
      return logval != a.logval;
      }
   bool operator>=(const logrealfast& a) const
      {
      return logval <= a.logval;
      }
   bool operator<=(const logrealfast& a) const
      {
      return logval >= a.logval;
      }
   bool operator>(const logrealfast& a) const
      {
      return logval < a.logval;
      }
   bool operator<(const logrealfast& a) const
      {
      return logval > a.logval;
      }
   // stream I/O
   friend std::ostream& operator<<(std::ostream& sout, const logrealfast& x);
   friend std::istream& operator>>(std::istream& sin, logrealfast& x);
   // specialized power function
   friend logrealfast pow(const logrealfast& a, const double b)
      {
      logrealfast result = a;
      result.logval *= b;
      logrealfast::ensurefinite(result.logval);
      return result;
      }
};

// private helper functions

inline void logrealfast::ensurefinite(double& x)
   {
   // trap infinity
   const int inf = isinf(x);
   if (inf < 0)
      {
#if DEBUG>=2
      std::cerr << "DEBUG (logrealfast): negative infinity." << std::endl;
#endif
      }
   else if (inf > 0)
      {
#if DEBUG>=2
      std::cerr << "DEBUG (logrealfast): positive infinity." << std::endl;
#endif
      }
   // trap NaN
   else if (isnan(x))
      {
      failwith("NaN cannot be represented");
      }
   }

// arithmetic operations - unary

/*! \brief add the value of 'b' to this value (let's call it 'a')
 * This method makes use of the equality:
 *      log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
 * which is derived from:
 *      a + b = a * (1 + (b / a))
 * It also makes use of the commutative property:
 *      log(a + b) = log(b + a)
 * by swapping a,b as needed, so that we only need to take the exponent of a
 * positive quantity.
 */
inline logrealfast& logrealfast::operator+=(const logrealfast& b)
   {
   static const double lutinvstep = (lutsize - 1) / lutrange;
   const double diff = fabs(b.logval - logval);

   if (b.logval < logval)
      logval = b.logval;

#if DEBUG>=3
   const double offset = log(1 + exp(-diff));
   //logval -= offset;
#endif

   if (diff < lutrange)
      {
      const int index = int(round(diff * lutinvstep));
      logval -= lut_add[index];
#if DEBUG>=3
      file << diff << "\t" << offset - lut_add[index] << std::endl;
#endif
      }
#if DEBUG>=3
   else
   file << diff << "\t" << offset << std::endl;
#endif

   return *this;
   }

/*! \brief subtract the value of 'b' from this value (let's call it 'a')
 * This method makes use of the equality:
 *      log(a - b) = log(a) + log(1 - exp(log(b) - log(a)))
 * which is derived from:
 *      a - b = a * (1 - (b / a))
 * Since this class cannot represent a negative number, the above is only
 * valid when a > b.
 */
inline logrealfast& logrealfast::operator-=(const logrealfast& b)
   {
   static const double lutinvstep = (lutsize - 1) / lutrange;
   const double diff = logval - b.logval;

   assertalways(logval > b.logval);

   if (diff < lutrange)
      {
      const int index = int(round(diff * lutinvstep));
      logval -= lut_sub[index];
      }

   return *this;
   }

// The following functions operate through the above - no need to make them friends

inline logrealfast operator+(const logrealfast& a, const logrealfast& b)
   {
   logrealfast result = a;
   result += b;
   return result;
   }

inline logrealfast operator-(const logrealfast& a, const logrealfast& b)
   {
   logrealfast result = a;
   result -= b;
   return result;
   }

inline logrealfast operator*(const logrealfast& a, const logrealfast& b)
   {
   logrealfast result = a;
   result *= b;
   return result;
   }

inline logrealfast operator/(const logrealfast& a, const logrealfast& b)
   {
   logrealfast result = a;
   result /= b;
   return result;
   }

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
