#ifndef __logrealfast_h
#define __logrealfast_h

#ifndef NDEBUG
//#define DEBUGFILE
#endif

#include "config.h"
#include "itfunc.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#ifdef DEBUGFILE
#  include <fstream>
#endif

namespace libbase {

/*!
   \brief   Fast Logarithm Arithmetic.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (21 Feb 2002)
  Fixed some bugs in the LUT working of the system; also added hooks to allow debugging
  by printing to a file the difference values and the errors for all LUT access.

   \version 1.11 (22 Feb 2002)
  Optimised the choice of LUT size and range (to 128k entries over [0,12]). Also slightly
  speeded up some other routines in minor ways.

   \version 1.12 (23 Feb 2002)
  Other minor speed enhancement changes.

   \version 1.13 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

   \version 1.14 (29 Mar 2002)
  modified constructor to trap infinite values and NaN; also moved constructor to
  implementation file, since it has become too large to be defined inline.

   \version 1.15 (4 Apr 2002)
  modified constructor to trap zero values first; since zero is the default argument, there
  are many more calls with this value than any other, so this should improve performance.

   \version 1.16 (4 Apr 2002)
  added default constructor to avoid going through the lengthy constructor for default.
  Also moved the LUT building code into a separate (private) function.

   \version 1.17 (6 Apr 2002)
  added assignment from double, as this would otherwise have to make use of the
  constructor, with an additional member copy. Also added a private conversion
  function that returns the logval representing a double, and modified the constructor
  to make use of that function. Also added copy constructors and copy assignment
  operators although the default (member-wise copy) will do anyway.

   \version 1.18 (15 Jun 2002)
  changed 'flags' variable in implementation file from type int to type
  ios::fmtflags, as it's supposed to be.

   \version 1.20 (19 Apr 2005)
  added specialized pow() function, to avoid conversion to double.

   \version 1.21 (17 Jul 2006)
  added explicit conversion of round's output to integer, following the change in
  itfunc 1.07

   \version 1.22 (28 Jul 2006)
  added a private helper function to perform the re-used operation where a double is
  checked for infinity and converted to a DBL_MAX of the correct sign. This makes it
  easier to monitor whenever this happens. Also made 'convertfromdouble' a static
  function.

   \version 1.23 (31 Jul 2006)
   - added comparison operators between variables of this kind - these are required by
  the turbo decoder when taking a hard decision (actually it only uses the greater-than
  operator, but all comparisons are defined here). When these were not supplied, the
  comparison was performed _after_ a conversion to double, which can easily cause
  under- or over-flow, leading to a useless comparison.
   - note that binary arithmetic operators were already provided.
   - added the binary minus operator as private, so no-one uses it.

   \version 1.30 (26 Oct 2006)
   - defined class and associated data within "libbase" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.31 (17 Jul 2007)
   - changed references to isinf() and isnan() back to global namespace, in accord with
    config.h 3.23.

   \version 1.32 (5 Nov 2007)
   - updated convertfromdouble() so that warnings are not repeated every time.
   - also changed warning messages to display on trace rather than clog.

   \version 1.33 (7 Nov 2007)
   - modified such that file output only occurs when DEBUGFILE is defined, rather
    than in all debug builds.
   - modified other debug warnings to occur when NDEBUG is not defined.

   \version 1.34 (13 Nov 2007)
   - updated ensurefinite() so that warnings stop showing if they are repeated often.
*/

class logrealfast {
   static const int  lutsize;
   static const double lutrange;
   static double  *lut;
   static bool    lutready;
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
   bool operator==(const logrealfast& a) const { return logval == a.logval; };
   bool operator>=(const logrealfast& a) const { return logval <= a.logval; };
   bool operator<=(const logrealfast& a) const { return logval >= a.logval; };
   bool operator>(const logrealfast& a) const { return logval < a.logval; };
   bool operator<(const logrealfast& a) const { return logval > a.logval; };
   // stream output
   friend std::ostream& operator<<(std::ostream& s, const logrealfast& x);
   // specialized power function
   friend logrealfast pow(const logrealfast& a, const double b);
};

// private helper functions

inline void logrealfast::ensurefinite(double& x)
   {
   const int inf = isinf(x);
   if(inf < 0)
      {
      x = -DBL_MAX;
#ifndef NDEBUG
      static int warningcount = 10;
      if(--warningcount > 0)
         trace << "WARNING (logrealfast): negative infinity.\n";
      else if(warningcount == 0)
         trace << "WARNING (logrealfast): last warning repeated too many times; stopped logging.\n";
#endif
      }
   else if(inf > 0)
      {
      x = DBL_MAX;
#ifndef NDEBUG
      static int warningcount = 10;
      if(--warningcount > 0)
         trace << "WARNING (logrealfast): positive infinity.\n";
      else if(warningcount == 0)
         trace << "WARNING (logrealfast): last warning repeated too many times; stopped logging.\n";
#endif
      }
   }

// construction operations

inline logrealfast::logrealfast()
   {
   if(!lutready)
      buildlut();
   }

inline logrealfast::logrealfast(const double m)
   {
   if(!lutready)
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
   static const double lutinvstep = (lutsize-1)/lutrange;
   const double diff = fabs(logval - a.logval);

   if(a.logval < logval)
      logval = a.logval;

#ifdef DEBUGFILE
   const double offset = log(1 + exp(-diff));
   logval -= offset;
#endif

   if(diff < lutrange)
      {
      const int index = int(round(diff*lutinvstep));
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

}; // end namespace

#endif
