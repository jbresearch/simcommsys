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

#ifndef __clonable_h
#define __clonable_h

#include "config.h"

namespace libbase {

/*!
 * \brief   Clonable class base.
 * \author  Johann Briffa
 *
 * All clonable classes inherit from this. Implements part of the required
 * functionality, in conjunction with DECLARE_CLONABLE() macro.
 */

class clonable {
public:
   virtual ~clonable()
      {
      }
   /*! \name Serialization Support */
   /*! \brief Cloning operation */
   virtual clonable *clone() const = 0;
   /* @} */
};

#define DECLARE_CLONABLE( class_name ) \
   /* Comment */ \
   private: \
   /*! \name Cloning Support */ \
   /*! \brief Heap creation function */ \
   static libbase::clonable* create() { return new class_name; } \
   /* @} */ \
   public: \
   libbase::clonable *clone() const { return new class_name(*this); }

} // end namespace

#endif
