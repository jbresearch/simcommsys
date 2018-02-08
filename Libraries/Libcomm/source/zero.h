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

#ifndef __source_zero_h
#define __source_zero_h

#include "config.h"
#include "source.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   All-zero source.
 * \author  Johann Briffa
 *
 * Implements a trivial source that always returns the zero symbol.
 */

template <class S, template <class > class C = libbase::vector>
class zero : public source<S, C> {
public:
   //! Generate a single source element
   S generate_single()
      {
      return 0;
      }

   //! Description
   std::string description() const
      {
      return "All-zero source";
      }

   // Serialization Support
DECLARE_SERIALIZER(zero)
};

} // end namespace

#endif

