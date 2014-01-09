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

#ifndef __elgamal_h
#define __elgamal_h

#include "config.h"
#include "equalitydisclog_zpschnorr.h"
#include "ciphertext.h"

#include <list>

namespace libbase {

/*!
 * \brief   ElGamal cipher
 * \author  Johann Briffa
 *
 * Main cipher class that performs encryption, decryption, generation
 * of decryption proofs and combining of shares.
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class elgamal {
public:
   static BigInteger decrypt(BigInteger p, BigInteger secretKey,
         BigInteger cipherText)
      {
      BigInteger partialDecryption = cipherText.pow_mod(secretKey, p);
      return partialDecryption;
      }

   static equalitydisclog_zpschnorr<BigInteger> createDecryptionProof(group<
         BigInteger> grp, BigInteger secretKey, BigInteger cipherText)
      {
      return equalitydisclog_zpschnorr<BigInteger>::constructProof(grp,
            grp.get_g(), cipherText, secretKey);
      }

   static BigInteger combineShares(std::list<BigInteger> shares,
         BigInteger cipherTextB, BigInteger p)
      {
      BigInteger plaintext(1);
      typedef typename std::list<BigInteger>::iterator iterator;
      for (iterator it = shares.begin(); it != shares.end(); it++)
         {
         plaintext = (plaintext * *it) % p;
         }
      plaintext = (cipherTextB * plaintext.inv_mod(p)) % p;
      return plaintext;
      }

   static ciphertext<BigInteger> encrypt(BigInteger plainText,
         group<BigInteger> grp, BigInteger pubKey)
      {
      BigInteger randomness;
      randomness.random(grp.get_q().size());

      BigInteger gr = grp.get_g().pow_mod(randomness, grp.get_p());
      BigInteger myr = (pubKey.pow_mod(randomness, grp.get_p()) * plainText)
            % grp.get_p();
      return ciphertext<BigInteger> (gr, myr);
      }
};

} // end namespace

#endif
