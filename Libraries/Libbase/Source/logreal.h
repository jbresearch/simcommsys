#ifndef __logreal_h
#define __logreal_h

#include "config.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>

namespace libbase {

/*!
   \brief   Logarithm Arithmetic.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.02 (15 Jun 2002)
  changed 'flags' variable in implementation file from type int to type
  ios::fmtflags, as it's supposed to be.

  Version 1.10 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.11 (17 Jul 2007)
  * changed references to isinf() and isnan() back to global namespace, in accord with
    config.h 3.23.
*/

class logreal {
   double       logval;
   logreal& operator-();
   logreal& operator-=(const logreal& a);
public:
   logreal(const double m=0);
   operator double() const;

   logreal& operator+=(const logreal& a);
   logreal& operator*=(const logreal& a);
   logreal& operator/=(const logreal& a);

   friend std::ostream& operator<<(std::ostream& s, const logreal& x);
};

inline logreal::logreal(const double m)
   {
   if(m < 0)
      {
      std::cerr << "FATAL ERROR (logreal): Negative numbers cannot be used.\n";
      exit(1);
      }
   logval = (m==0) ? DBL_MAX : -log(m);
   }

inline logreal::operator double() const
   {
   return exp(-logval);
   }
   
inline logreal& logreal::operator+=(const logreal& a)
   {
   if(logval < a.logval)
      logval = logval - log(1 + exp(logval - a.logval));
   else
      logval = a.logval - log(1 + exp(a.logval - logval));
   return *this;
   }
   
inline logreal& logreal::operator*=(const logreal& a)
   {
   logval += a.logval;
   if(isinf(logval))
      logval = isinf(logval) * DBL_MAX;
   return *this;
   }
   
inline logreal& logreal::operator/=(const logreal& a)
   {
   logval -= a.logval;
   if(isinf(logval))
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

}; // end namespace

#endif
