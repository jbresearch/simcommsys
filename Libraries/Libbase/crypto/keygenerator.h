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

#ifndef __keygenerator_h
#define __keygenerator_h

#include "config.h"
#include "group.h"
#include "keypair.h"

#include <list>

namespace libbase {

/*!
 * \brief   Utility class to manipulate key shares
 * \author  Johann Briffa
 *
 * Utility class to generate key shares and combine the shares into
 * a single public key.
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class keygenerator {
public:
   static keypair<BigInteger> generateKeyShare(group<BigInteger> grp)
      {
      BigInteger secretShare = group<BigInteger>::get_random_integer(
            grp.get_q());
      BigInteger publicShare = grp.get_g().pow_mod(secretShare, grp.get_p());
      return keypair<BigInteger> (secretShare, publicShare);
      }

   static BigInteger combinePublicKeyShares(std::list<BigInteger> pubKeys,
         group<BigInteger> grp)
      {
      //G combinedPK = grp.zero() ;
      BigInteger combinedPubKey(1);
      typedef typename std::list<BigInteger>::iterator iterator;
      for (iterator it = pubKeys.begin(); it != pubKeys.end(); it++)
         {
         combinedPubKey = (combinedPubKey * *it) % grp.get_p();
         }
      return combinedPubKey;
      }
};

} // end namespace

#endif
