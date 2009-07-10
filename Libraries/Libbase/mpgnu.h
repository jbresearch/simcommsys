#ifndef __mpgnu_h
#define __mpgnu_h

#include "config.h"
#include <float.h>
#include <iostream>

#ifdef GMP
#include <gmp.h>
#endif

namespace libbase {

/*!
 \brief   GNU Multi-Precision Arithmetic.
 \author  Johann Briffa

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 \version 1.01 (6 Mar 2002)
 changed vcs version variable from a global to a static class variable.
 also changed use of iostream from global to std namespace.

 \version 1.10 (26 Oct 2006)
 - defined class and associated data within "libbase" namespace.
 - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class mpgnu {
   static void init();
#ifdef GMP
   static mpf_t dblmin, dblmax;
   mpf_t value;
#endif
public:
   ~mpgnu();
   mpgnu(const double m = 0);
   mpgnu(const mpgnu& a);

   operator double() const;

   mpgnu& operator=(const mpgnu& a);

   mpgnu& operator-();
   mpgnu& operator+=(const mpgnu& a);
   mpgnu& operator-=(const mpgnu& a);
   mpgnu& operator*=(const mpgnu& a);
   mpgnu& operator/=(const mpgnu& a);

   friend mpgnu operator+(const mpgnu& a, const mpgnu& b);
   friend mpgnu operator-(const mpgnu& a, const mpgnu& b);
   friend mpgnu operator*(const mpgnu& a, const mpgnu& b);
   friend mpgnu operator/(const mpgnu& a, const mpgnu& b);

   friend std::ostream& operator<<(std::ostream& s, const mpgnu& x);
};

// Initialisation / Destruction

inline mpgnu::~mpgnu()
   {
#ifdef GMP
   mpf_clear(value);
#endif
   }

inline mpgnu::mpgnu(const double m)
   {
   init();
#ifdef GMP
   mpf_init2(value, 256);
   mpf_set_d(value, m);
#endif
   }

inline mpgnu::mpgnu(const mpgnu& a)
   {
   init();
#ifdef GMP
   mpf_init2(value, 256);
   mpf_set(value, a.value);
#endif
   }

// Conversion

inline mpgnu::operator double() const
   {
#ifndef GMP
   double result = 0;
#else
   double result;
   if(mpf_cmp(value, dblmin) <= 0)
   result = DBL_MIN;
   else if(mpf_cmp(value, dblmax) >= 0)
   result = DBL_MAX;
   else
   result = mpf_get_d(value);
#endif
   return result;
   }

inline mpgnu& mpgnu::operator=(const mpgnu& a)
   {
#ifdef GMP
   mpf_set(value, a.value);
#endif
   return *this;
   }

// Base Operations

inline mpgnu& mpgnu::operator-()
   {
#ifdef GMP
   mpf_neg(value, value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator+=(const mpgnu& a)
   {
#ifdef GMP
   mpf_add(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator-=(const mpgnu& a)
   {
#ifdef GMP
   mpf_sub(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator*=(const mpgnu& a)
   {
#ifdef GMP
   mpf_mul(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator/=(const mpgnu& a)
   {
#ifdef GMP
   mpf_div(value, value, a.value);
#endif
   return *this;
   }

// Derived Operations (Friends)

inline mpgnu operator+(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef GMP
   mpf_add(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator-(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef GMP
   mpf_sub(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator*(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef GMP
   mpf_mul(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator/(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef GMP
   mpf_div(result.value, a.value, b.value);
#endif
   return result;
   }

// Input/Output Operations

inline std::ostream& operator<<(std::ostream& s, const mpgnu& x)
   {
#ifdef GMP
   using std::ios;

   int flags = s.flags();
   s.setf(ios::fixed, ios::floatfield);

   const int digits = 6;
   mp_exp_t exponent;
   char mantissa[digits+2];
   mpf_get_str(mantissa, &exponent, 10, digits, x.value);
   s << "0." << mantissa;
   s.setf(ios::showpos);
   s << "e" << exponent;

   s.flags(flags);
#endif
   return s;
   }

} // end namespace

#endif
