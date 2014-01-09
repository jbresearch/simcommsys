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

#ifndef __stream_lut_h
#define __stream_lut_h

#include "config.h"
#include "interleaver/lut/named_lut.h"
#include <cstdio>

namespace libcomm {

/*!
 * \brief   Stream-loaded LUT Interleaver.
 * \author  Johann Briffa
 *
 */

template <class real>
class stream_lut : public named_lut<real> {
public:
   stream_lut(const char *filename, FILE *file, const int tau, const int m);
   ~stream_lut()
      {
      }
};

} // end namespace

#endif
