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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __itfunc_h
#define __itfunc_h

#include "config.h"
#include "matrix.h"
#include <cmath>
#include <cstdlib>
#include <string>

/*!
 * \file
 * \brief   Information Theory Functions.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

namespace libbase {

/*! \name Gamma functions */
//! \sa Numerical Recipes in C, p.216-219
double gammln(double xx);
double gammser(double a, double x);
double gammcf(double a, double x);
double gammp(double a, double x);
// @}

/*! \name Error function */
double cerf(double x);
double cerfc(double x);
/*! \brief Error function based on incomplete Gamma functions
 * More accurate but much slower than Chebychev fitting
 */
inline double erff(double x)
   {
   return x < 0.0 ? -gammp(0.5, x * x) : gammp(0.5, x * x);
   }
/*! \brief Complementary rrror function based on incomplete Gamma functions
 * More accurate but much slower than Chebychev fitting
 */
inline double erffc(double x)
   {
   return x < 0.0 ? 1.0 + gammp(0.5, x * x) : 1.0 - gammp(0.5, x * x);
   }
// @}

inline double Q(double x)
   {
   return 0.5 * erffc(x / sqrt(2.0));
   }
inline double gauss(double x)
   {
   return exp(-0.5 * x * x) / sqrt(2.0 * PI);
   }

//! Limits numbers between a high and low limit
template <class T>
inline T limit(const T x, const T lo, const T hi)
   {
   return std::max(lo, std::min(hi, x));
   }

int weight(int cw);

/*! \brief Binary Hamming weight
 */
template <class T>
int weight(const matrix<T>& m)
   {
   matrix<int> t;
   t = m;
   t.apply(weight);
   return t.sum();
   }

//! Gray code
inline int32u gray(int32u n)
   {
   return n ^ (n >> 1);
   }
int32u igray(int32u n);

int gcd(int a, int b);

std::string hexify(const std::string input);
std::string dehexify(const std::string input);

} // end namespace

#endif
