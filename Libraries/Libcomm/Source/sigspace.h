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
*/

namespace libcomm {

class sigspace {
   static const libbase::vcs version;
   double	inphase, quad;
public:
   sigspace(const double i=0, const double q=0);

   double i() const { return inphase; };
   double q() const { return quad; };
   double r() const { return sqrt(i()*i() + q()*q()); };
   double p() const { return atan2(q(), i()); };
   operator double() const { return r(); };

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

   friend std::ostream& operator<<(std::ostream& s, const sigspace& x);
   friend std::istream& operator>>(std::istream& s, sigspace& x);
};

}; // end namespace

#endif
