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

#ifndef __ciphertext_h
#define __ciphertext_h

#include "config.h"

namespace libbase {

/*!
 * \brief   The two parts of the Ciphertext
 * \author  Johann Briffa
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class ciphertext {
private:
   BigInteger gr;
   BigInteger myr;

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   explicit ciphertext(BigInteger gr = 0, BigInteger myr = 0) :
      gr(gr), myr(myr)
      {
      }
   // @}

   /*! \name Getters */
   const BigInteger& get_gr() const
      {
      return gr;
      }
   const BigInteger& get_myr() const
      {
      return myr;
      }
   // @}
};

} // end namespace

#endif
