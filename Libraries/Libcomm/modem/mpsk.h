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

#ifndef __mpsk_h
#define __mpsk_h

#include "config.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   M-PSK Modulator.
 * \author  Johann Briffa
 *
 * \note Gray code mapping is used for binary representation of
 * adjacent points on the constellation.
 */

class mpsk : public lut_modulator {
protected:
   mpsk()
      {
      }
   void init(const int m);
public:
   mpsk(const int m)
      {
      init(m);
      }
   ~mpsk()
      {
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(mpsk)
};

} // end namespace

#endif
