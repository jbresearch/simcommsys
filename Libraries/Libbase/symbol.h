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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __symbol_h
#define __symbol_h

#include "config.h"

namespace libcomm {

/*!
 * \brief   Finite symbol base class.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Created to abstract the concept of a symbol from a finite alphabet.
 * This is an abstract class which defines the interface to such an object.
 */

class symbol {
public:
   /*! \name Constructors / Destructors */
   virtual ~symbol() = 0;
   // @}

   /*! \name Type conversion */
   virtual operator int() const = 0;
   virtual symbol& operator=(const int value) = 0;
   // @}

   /*! \name Class parameters */
   //! Number of elements in the finite alphabet
   virtual int elements() const = 0;
   // @}
};

/*!
 * \brief   Finite q-ary symbol.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Uses an integer to represent symbol value; value is initialized to zero
 * on creation.
 */

template <int q>
class finite_symbol {
private:
   /*! \name Object representation */
   //! Representation of this element by its polynomial coefficients
   int value;
   // @}

private:
   /*! \name Internal functions */
   void init(int value);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   explicit finite_symbol(int value = 0)
      {
      init(value);
      }
   ~finite_symbol()
      {
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
   int elements() const
      {
      return q;
      }
   // @}
};

} // end namespace

#endif
