#ifndef __sigspace_h
#define __sigspace_h
      
#include "config.h"
#include "vcs.h"
#include <iostream.h>
#include <math.h>

extern const vcs sigspace_version;

class sigspace {
   double	inphase, quad;
public:
   sigspace(const double i=0, const double q=0);
   double i() const;
   double q() const;
   double r() const;
   double p() const;
   operator double() const;
   sigspace& operator+=(const sigspace& a);
   sigspace& operator-=(const sigspace& a);
   sigspace& operator*=(const double& a);
   sigspace& operator/=(const double& a);
   friend sigspace operator+(const sigspace& a, const sigspace& b);
   friend sigspace operator-(const sigspace& a, const sigspace& b);
   friend sigspace operator*(const sigspace& a, const double& b);
   friend sigspace operator/(const sigspace& a, const double& b);
   friend ostream& operator<<(ostream& s, const sigspace& x);
   friend istream& operator>>(istream& s, sigspace& x);
};

inline sigspace::sigspace(const double i, const double q)
   {
   inphase = i;
   quad = q;
   }   
   
inline double sigspace::i() const
   {
   return inphase;
   }
   
inline double sigspace::q() const
   {
   return quad;
   }

inline double sigspace::r() const
   {
   return sqrt(i()*i() + q()*q());
   }

inline double sigspace::p() const
   {
   return atan2(q(), i());
   }

inline sigspace::operator double() const
   {
   return r();
   }
   
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

inline sigspace& sigspace::operator*=(const double& a)
   {
   inphase *= a;
   quad *= a;
   return *this;
   }

inline sigspace& sigspace::operator/=(const double& a)
   {
   inphase /= a;
   quad /= a;
   return *this;
   }

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
   
inline sigspace operator*(const sigspace& a, const double& b)
   {
   sigspace c = a;
   c *= b;
   return c;
   }
   
inline sigspace operator/(const sigspace& a, const double& b)
   {
   sigspace c = a;
   c /= b;
   return c;
   }

#endif
