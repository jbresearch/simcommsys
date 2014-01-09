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

#ifndef __mpgnu_h
#define __mpgnu_h

#include "config.h"
#include <cfloat>
#include <iostream>

#ifdef USE_GMP
#include <gmp.h>
#endif

namespace libbase {

/*!
 * \brief   GNU Multi-Precision Arithmetic.
 * \author  Johann Briffa
 *
 * \version 1.01 (6 Mar 2002)
 * changed vcs version variable from a global to a static class variable.
 * also changed use of iostream from global to std namespace.
 *
 * \version 1.10 (26 Oct 2006)
 * - defined class and associated data within "libbase" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class mpgnu {
   static void init();
#ifdef USE_GMP
   static mpf_t dblmin, dblmax;
   mpf_t value;
#endif
public:
   ~mpgnu();
   mpgnu();
   mpgnu(const mpgnu& a);
   mpgnu(const double a);

   operator double() const;

   mpgnu& operator=(const mpgnu& a);
   mpgnu& operator=(const double a);

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
   friend std::istream& operator>>(std::istream& s, mpgnu& x);
};

// Initialisation / Destruction

inline mpgnu::~mpgnu()
   {
#ifdef USE_GMP
   mpf_clear(value);
#endif
   }

inline mpgnu::mpgnu()
   {
   init();
#ifdef USE_GMP
   mpf_init2(value, 256);
#endif
   }

inline mpgnu::mpgnu(const mpgnu& a)
   {
   init();
#ifdef USE_GMP
   mpf_init2(value, 256);
   mpf_set(value, a.value);
#endif
   }

inline mpgnu::mpgnu(const double a)
   {
   init();
#ifdef USE_GMP
   mpf_init2(value, 256);
   mpf_set_d(value, a);
#endif
   }

// Conversion

inline mpgnu::operator double() const
   {
#ifndef USE_GMP
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
#ifdef USE_GMP
   mpf_set(value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator=(const double a)
   {
#ifdef USE_GMP
   mpf_set_d(value, a);
#endif
   return *this;
   }

// Base Operations

inline mpgnu& mpgnu::operator-()
   {
#ifdef USE_GMP
   mpf_neg(value, value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator+=(const mpgnu& a)
   {
#ifdef USE_GMP
   mpf_add(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator-=(const mpgnu& a)
   {
#ifdef USE_GMP
   mpf_sub(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator*=(const mpgnu& a)
   {
#ifdef USE_GMP
   mpf_mul(value, value, a.value);
#endif
   return *this;
   }

inline mpgnu& mpgnu::operator/=(const mpgnu& a)
   {
#ifdef USE_GMP
   mpf_div(value, value, a.value);
#endif
   return *this;
   }

// Derived Operations (Friends)

inline mpgnu operator+(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef USE_GMP
   mpf_add(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator-(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef USE_GMP
   mpf_sub(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator*(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef USE_GMP
   mpf_mul(result.value, a.value, b.value);
#endif
   return result;
   }

inline mpgnu operator/(const mpgnu& a, const mpgnu& b)
   {
   mpgnu result;
#ifdef USE_GMP
   mpf_div(result.value, a.value, b.value);
#endif
   return result;
   }

// Input/Output Operations

inline std::ostream& operator<<(std::ostream& s, const mpgnu& x)
   {
#ifdef USE_GMP
   using std::ios;

   const ios::fmtflags flags = s.flags();
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

inline std::istream& operator>>(std::istream& s, mpgnu& x)
   {
#ifdef USE_GMP
   std::string str;
   s >> str;

   mpf_set_str(x.value, str.c_str(), 10);
#endif
   return s;
   }

} // end namespace

#endif
