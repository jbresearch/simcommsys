#ifndef __sigspace_h
#define __sigspace_h
      
#include "config.h"
#include "vcs.h"
#include <math.h>
#include <iostream>

/*
  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.02 (13 Mar 2002)
  moved most functions to the implementation file instead of here.

  Version 1.10 (27 Mar 2002)
  added two functions to make multiplication with double commutative. Also, made
  passing of all double parameters direct, not by reference.

  Version 1.20 (30 Oct 2006)
  * defined class and associated data within "libcomm" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.

  Version 1.30 (14 Nov 2007)
  * added equality and inequality operators.
  * moved most functions here and made them inline.
*/

namespace libcomm {

class sigspace {
   static const libbase::vcs version;
   double	inphase, quad;
public:
   // creator / destructor
   sigspace(const double i=0, const double q=0);

   double i() const { return inphase; };
   double q() const { return quad; };
   double r() const { return sqrt(i()*i() + q()*q()); };
   double p() const { return atan2(q(), i()); };
   operator double() const { return r(); };

   // comparison operations
   bool operator==(const sigspace& a);
   bool operator!=(const sigspace& a);

   // arithmetic operations
   sigspace& operator+=(const sigspace& a);
   sigspace& operator-=(const sigspace& a);
   sigspace& operator*=(const double a);
   sigspace& operator/=(const double a);
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

// comparison operations

inline bool sigspace::operator==(const sigspace& a)
   {
   return(inphase == a.inphase && quad == a.quad);
   }

inline bool sigspace::operator!=(const sigspace& a)
   {
   return(inphase != a.inphase || quad != a.quad);
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

}; // end namespace

#endif
