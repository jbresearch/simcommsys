#ifndef __sigspace_h
#define __sigspace_h

#include "config.h"
#include <math.h>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Signal Space point.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
   sigspace(const double i = 0, const double q = 0);

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
   sigspace& operator+=(const sigspace& a);
   sigspace& operator-=(const sigspace& a);
   sigspace& operator*=(const double a);
   sigspace& operator/=(const double a);
   // arithmetic operations - friends
   friend bool operator==(const sigspace& a, const sigspace& b);
   friend bool operator!=(const sigspace& a, const sigspace& b);
   friend sigspace operator-(const sigspace& a);
   friend sigspace operator+(const sigspace& a, const sigspace& b);
   friend sigspace operator-(const sigspace& a, const sigspace& b);
   friend sigspace operator*(const sigspace& a, const double b);
   friend sigspace operator/(const sigspace& a, const double b);
   friend sigspace operator*(const double a, const sigspace& b);
   friend sigspace operator/(const double a, const sigspace& b);

   // stream input / output
   friend std::ostream& operator<<(std::ostream& s, const sigspace& x);
   friend std::istream& operator>>(std::istream& s, sigspace& x);
};

// creator / destructor

inline sigspace::sigspace(const double i, const double q)
   {
   inphase = i;
   quad = q;
   }

// arithmetic operations

inline sigspace& sigspace::operator+=(const sigspace& a)
   {
   inphase += a.inphase;
   quad += a.quad;
   return *this;
   }

inline sigspace& sigspace::operator-=(const sigspace& a)
   {
   inphase -= a.inphase;
   quad -= a.quad;
   return *this;
   }

inline sigspace& sigspace::operator*=(const double a)
   {
   inphase *= a;
   quad *= a;
   return *this;
   }

inline sigspace& sigspace::operator/=(const double a)
   {
   inphase /= a;
   quad /= a;
   return *this;
   }

// comparison operations

inline bool operator==(const sigspace& a, const sigspace& b)
   {
   return (a.inphase == b.inphase && a.quad == b.quad);
   }

inline bool operator!=(const sigspace& a, const sigspace& b)
   {
   return (a.inphase != b.inphase || a.quad != b.quad);
   }

// arithmetic operations - unary

inline sigspace operator-(const sigspace& a)
   {
   return sigspace(-a.inphase, -a.quad);
   }

// arithmetic operations - friends

inline sigspace operator+(const sigspace& a, const sigspace& b)
   {
   sigspace c = a;
   c += b;
   return c;
   }

inline sigspace operator-(const sigspace& a, const sigspace& b)
   {
   sigspace c = a;
   c -= b;
   return c;
   }

inline sigspace operator*(const sigspace& a, const double b)
   {
   sigspace c = a;
   c *= b;
   return c;
   }

inline sigspace operator/(const sigspace& a, const double b)
   {
   sigspace c = a;
   c /= b;
   return c;
   }

inline sigspace operator*(const double a, const sigspace& b)
   {
   sigspace c = b;
   c *= a;
   return c;
   }

inline sigspace operator/(const double a, const sigspace& b)
   {
   sigspace c = b;
   c /= a;
   return c;
   }

} // end namespace

#endif
