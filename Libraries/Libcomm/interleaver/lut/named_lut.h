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

#ifndef __named_lut_h
#define __named_lut_h

#include "config.h"
#include "interleaver/lut_interleaver.h"
#include "serializer.h"
#include <string>
#include <iostream>

namespace libcomm {

/*!
 * \brief   Named LUT Interleaver.
 * \author  Johann Briffa
 *
 * Implements an interleaver which is specified directly by its LUT,
 * and which is externally generated (say by Simulated Annealing
 * or another such method).
 * A name is associated with the interleaver (say, filename).
 */

template <class real>
class named_lut : public lut_interleaver<real> {
protected:
   std::string lutname;
   int m;
   named_lut()
      {
      }
public:
   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(named_lut)
};

} // end namespace

#endif

