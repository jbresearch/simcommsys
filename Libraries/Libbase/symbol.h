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

#ifndef __symbol_h
#define __symbol_h

#include "config.h"

namespace libbase {

/*!
 * \brief   Finite q-ary symbol.
 * \author  Johann Briffa
 *
 * Implements the concept of a symbol from a finite alphabet.
 * Uses an integer to represent symbol value; value is initialized to zero
 * on creation.
 */

template <int q>
class symbol {
private:
   /*! \name Object representation */
   //! Representation of this element as an index into the alphabet
   int value;
   // @}

private:
   /*! \name Internal functions */
   /*!
    * \brief Initialization
    * \param   value Integer representation of element
    */
   void init(int value)
      {
      assert(value >= 0 && value < q);
      this->value = value;
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   explicit symbol(int value = 0)
      {
      init(value);
      }
   // @}

   /*! \name Type conversion */
   operator int() const
      {
      return value;
      }
   symbol& operator=(const int value)
      {
      init(value);
      return *this;
      }
   // @}

   /*! \name Class parameters */
   //! Number of elements in the finite alphabet
   static int elements()
      {
      return q;
      }
   // @}
};

} // end namespace


// Pre-processor sequence for explicit instantiations

#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#define SYMBOL_TYPE(z, n, text) \
   (symbol<n>)

#define SYMBOL_TYPE_SEQ \
   BOOST_PP_REPEAT_FROM_TO(2, 101, SYMBOL_TYPE, _)

#endif
