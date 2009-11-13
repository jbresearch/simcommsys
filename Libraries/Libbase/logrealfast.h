#ifndef __logrealfast_h
#define __logrealfast_h

#ifndef NDEBUG
//#define DEBUGFILE
#endif

#include "config.h"
#include "itfunc.h"
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#ifdef DEBUGFILE
#  include <fstream>
#endif

namespace libbase {

/*!
 * \brief   Fast Logarithm Arithmetic.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \note There is a hook to allow debugging by printing to a file the
 * difference values and the errors for all LUT access.
 * To activate, define DEBUGFILE.
 *
 * Implements log-scale arithmetic with table-lookup for speeding up
 * addition. The choice of LUT size and range is optimized at 128k entries
 * over [0,12].
 *
 * \note Constructor traps infinite values and NaN. Zero values are trapped
 * first; since zero is the default argument, there are many more calls
 * with this value than any other, so this should improve performance.
 *
 *
 * \note Comparison operators are provided between variables of this kind -
 * these are required by the turbo decoder when taking a hard decision
 * (actually it only uses the greater-than operator, but all comparisons
 * are defined here). When these were not supplied, the comparison was
 * performed _after_ a conversion to double, which can easily cause
 * under- or over-flow, leading to a useless comparison.
 */

class logrealfast {
   static const int lutsize;
   static const double lutrange;
   static double *lut;
   static bool lutready;
#ifdef DEBUGFILE
   static std::ofstream file;
#endif
private:
   double logval;
   void buildlut();
   static double convertfromdouble(const double m);
   static void ensurefinite(double& x);
   // define these as private to ensure no-one uses them
   logrealfast& operator-();
   logrealfast& operator-=(const logrealfast& a);
   logrealfast& operator-(const logrealfast& a) const;
public:
   // construction
   logrealfast();
   logrealfast(const double m);
   logrealfast(const logrealfast& a);
   // copy assignment
   logrealfast& operator=(const logrealfast& a);
   // conversion
   operator double() const;
   logrealfast& operator=(const double m);
   // arithmetic - unary
   logrealfast& operator+=(const logrealfast& a);
   logrealfast& operator*=(const logrealfast& a);
   logrealfast& operator/=(const logrealfast& a);
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
   friend logrealfast pow(const logrealfast& a, const double b);
};

// private helper functions

inline void logrealfast::ensurefinite(double& x)
   {
   const int inf = isinf(x);
   if (inf < 0)
      {
      x = -DBL_MAX;
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace << "WARNING (logrealfast): negative infinity.\n";
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging.\n";
#endif
      }
   else if (inf > 0)
      {
      x = DBL_MAX;
#ifndef NDEBUG
      static int warningcount = 10;
      if (--warningcount > 0)
         trace << "WARNING (logrealfast): positive infinity.\n";
      else if (warningcount == 0)
         trace
               << "WARNING (logrealfast): last warning repeated too many times; stopped logging.\n";
#endif
      }
   }

// construction operations

inline logrealfast::logrealfast()
   {
   if (!lutready)
      buildlut();
   }

inline logrealfast::logrealfast(const double m)
   {
   if (!lutready)
      buildlut();
   logval = convertfromdouble(m);
   }

inline logrealfast::logrealfast(const logrealfast& a)
   {
   // copy constructor need not check for lutready since at least one object
   // must have been created already.
   logval = a.logval;
   }

// copy assignment

inline logrealfast& logrealfast::operator=(const logrealfast& a)
   {
   logval = a.logval;
   return *this;
   }

// conversion operations

inline logrealfast::operator double() const
   {
   return exp(-logval);
   }

inline logrealfast& logrealfast::operator=(const double m)
   {
   logval = convertfromdouble(m);
   return *this;
   }

// arithmetic operations - unary

inline logrealfast& logrealfast::operator+=(const logrealfast& a)
   {
   static const double lutinvstep = (lutsize - 1) / lutrange;
   const double diff = fabs(logval - a.logval);

   if (a.logval < logval)
      logval = a.logval;

#ifdef DEBUGFILE
   const double offset = log(1 + exp(-diff));
   logval -= offset;
#endif

   if (diff < lutrange)
      {
      const int index = int(round(diff * lutinvstep));
      logval -= lut[index];
#ifdef DEBUGFILE
      file << diff << "\t" << offset - lut[index] << "\n";
#endif
      }
#ifdef DEBUGFILE
   else
   file << diff << "\t" << offset << "\n";
#endif

   return *this;
   }

inline logrealfast& logrealfast::operator*=(const logrealfast& a)
   {
   logval += a.logval;
   ensurefinite(logval);
   return *this;
   }

inline logrealfast& logrealfast::operator/=(const logrealfast& a)
   {
   logval -= a.logval;
   ensurefinite(logval);
   return *this;
   }

// The following functions operate through the above - no need to make them friends

inline logrealfast operator+(const logrealfast& a, const logrealfast& b)
   {
   logrealfast result = a;
   result += b;
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

// specialized power function

inline logrealfast pow(const logrealfast& a, const double b)
   {
   logrealfast result = a;
   result.logval *= b;
   logrealfast::ensurefinite(result.logval);
   return result;
   }

} // end namespace

#endif
