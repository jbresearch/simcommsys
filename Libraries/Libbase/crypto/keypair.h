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

#ifndef __keypair_h
#define __keypair_h

#include "config.h"

namespace libbase {

/*!
 * \brief   Utility class to hold public and private key
 * \author  Johann Briffa
 *
 * Also contains methods for saving and loading keys.
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class keypair {
private:
   BigInteger pri_key;
   BigInteger pub_key;

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   keypair()
      {
      }
   explicit keypair(BigInteger pri_key, BigInteger pub_key) :
      pri_key(pri_key), pub_key(pub_key)
      {
      }
   // @}

   /*! \name Getters */
   const BigInteger& get_pri_key() const
      {
      return pri_key;
      }
   const BigInteger& get_pub_key() const
      {
      return pub_key;
      }
   // @}

   /*! \name Stream I/O */
   friend std::ostream& operator<<(std::ostream& sout, const keypair& x)
      {
      sout << x.pri_key << std::endl;
      sout << x.pub_key << std::endl;
      return sout;
      }

   friend std::istream& operator>>(std::istream& sin, keypair& x)
      {
      sin >> x.pri_key;
      sin >> x.pub_key;
      return sin;
      }
   // @}
};

} // end namespace

#endif
