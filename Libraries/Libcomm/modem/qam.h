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

#ifndef __qam_h
#define __qam_h

#include "config.h"
#include "lut_modulator.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   QAM Modulator.
 * \author  Johann Briffa
 *
 * \version 1.00 (3 Jan 2008)
 * - Initial version, implements square QAM with Gray-coded mapping
 * - Derived from mpsk 2.20
 */

class qam : public lut_modulator {
protected:
   /*! \name Internal operations */
   void init(const int m);
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   qam()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   qam(const int m)
      {
      init(m);
      }
   ~qam()
      {
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qam)
};

} // end namespace

#endif
