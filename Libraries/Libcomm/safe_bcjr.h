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

#ifndef __safe_bcjr_h
#define __safe_bcjr_h

#include "bcjr.h"

namespace libcomm {

/*!
 * \brief   Safe version of BCJR - Standard template (no normalization).
 * \author  Johann Briffa
 *
 */

template <class real, class dbl = double>
class safe_bcjr : public bcjr<real, dbl> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<real, dbl> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<real, dbl> (encoder, tau)
      {
      }
};

/*!
 * \brief   Safe version of BCJR - 'double' specialization (normalized).
 * \author  Johann Briffa
 *
 */

template <>
class safe_bcjr<double, double> : public bcjr<double, double, true> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<double, double, true> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<double, double, true> (encoder, tau)
      {
      }
};

/*!
 * \brief   Safe version of BCJR - 'float' specialization (normalized).
 * \author  Johann Briffa
 *
 */

template <>
class safe_bcjr<float, float> : public bcjr<float, float, true> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<float, float, true> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<float, float, true> (encoder, tau)
      {
      }
};

/*!
 * \brief   Safe version of BCJR - 'float' specialization with 'double'
 * interface (normalized).
 * \author  Johann Briffa
 *
 */

template <>
class safe_bcjr<float, double> : public bcjr<float, double, true> {
protected:
   // default constructor
   safe_bcjr() :
      bcjr<float, double, true> ()
      {
      }
public:
   // constructor & destructor
   safe_bcjr(fsm& encoder, const int tau) :
      bcjr<float, double, true> (encoder, tau)
      {
      }
};

} // end namespace

#endif

