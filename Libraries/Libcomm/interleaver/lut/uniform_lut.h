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

#ifndef __uniform_lut_h
#define __uniform_lut_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"
#include "randgen.h"
#include <iostream>

namespace libcomm {

/*!
 * \brief   Uniform Interleaver.
 * \author  Johann Briffa
 *
 * \note This interleaver allows JPL termination.
 */

template <class real>
class uniform_lut : public lut_interleaver<real> {
   libbase::randgen r;
   int tau, m;
protected:
   void init(const int tau, const int m);
   uniform_lut()
      {
      }
public:
   uniform_lut(const int tau, const int m)
      {
      init(tau, m);
      }
   ~uniform_lut()
      {
      }

   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }
   void advance();

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(uniform_lut)
};

} // end namespace

#endif
