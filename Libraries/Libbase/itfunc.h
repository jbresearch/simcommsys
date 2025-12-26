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

#ifndef __itfunc_h
#define __itfunc_h

#include "config.h"
#include "matrix.h"
#include <cmath>
#include <cstdlib>
#include <string>

#include <boost/math/special_functions/erf.hpp>

/*!
 * \file
 * \brief   Information Theory Functions.
 * \author  Johann Briffa
 *
 */

namespace libbase
{

//! Tail probability of the standard normal distribution
template <class T>
inline T
Q(T x)
{
    return 0.5 * boost::math::erfc(x / sqrt(2.0));
}

//! Inverse tail probability of the standard normal distribution
template <class T>
inline T
Qinv(T y)
{
    return sqrt(2.0) * boost::math::erfc_inv(2 * y);
}

template <class T>
inline T
gauss(T x)
{
    return exp(-0.5 * x * x) / sqrt(2.0 * PI);
}

//! Limits numbers between a high and low limit
template <class T>
inline T
limit(const T x, const T lo, const T hi)
{
    return std::max(lo, std::min(hi, x));
}

int weight(int cw);

/*! \brief Binary Hamming weight
 */
template <class T>
int
weight(const matrix<T>& m)
{
    matrix<int> t;
    t = m;
    t.apply(weight);
    return t.sum();
}

//! Gray code
inline uint32_t
gray(uint32_t n)
{
    return n ^ (n >> 1);
}
uint32_t igray(uint32_t n);

int gcd(int a, int b);

std::string hexify(const std::string input);
std::string dehexify(const std::string input);

} // namespace libbase

#endif
