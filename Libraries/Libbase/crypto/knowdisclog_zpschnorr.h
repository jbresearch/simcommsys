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

#ifndef __knowdisclog_zpschnorr_h
#define __knowdisclog_zpschnorr_h

#include "config.h"
#include "group.h"
#include "keygenerator.h"
#include "sha.h"

namespace libbase {

/*!
 * \brief   Log-domain knowledge proof
 * \author  Johann Briffa
 *
 * Proof that an entity knows x in v = g^x.
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class knowdisclog_zpschnorr {
private:
   BigInteger a;
   BigInteger c;
   BigInteger r;
   BigInteger v;

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   explicit knowdisclog_zpschnorr(BigInteger a, BigInteger c, BigInteger r,
         BigInteger v) :
      a(a), c(c), r(r), v(v)
      {
      }
   // @}

   /* \brief Proof construction
    *
    * p,q,g are the parameters of the ElGamal (not included here).
    * z = random in Z_q
    * a = g^z mod p
    * c = hash(v,a)
    * r = (z + cx) mod q
    */
   static knowdisclog_zpschnorr createProof(group<BigInteger> grp,
         BigInteger x, BigInteger v)
      {
      BigInteger g = grp.get_g();
      BigInteger z = group<BigInteger>::get_random_integer(grp.get_q());
      BigInteger a = g.pow_mod(z, grp.get_p());

      std::vector<unsigned char> buf;
      buf += v.bytearray();
      buf += a.bytearray();

      libcomm::sha md;
      md.process(buf);

      BigInteger c(md);
      c %= grp.get_q();

      BigInteger r = (z + (c * x)) % grp.get_q();

      return knowdisclog_zpschnorr(a, c, r, v);
      }

   /* \brief Proof verification
    *
    * To verify proof, check that g^r = av^c (mod p)
    */
   static bool verifyProof(group<BigInteger> grp, knowdisclog_zpschnorr E)
      {
      BigInteger u = grp.get_g().pow_mod(E.r, grp.get_p());
      BigInteger w = (E.v.pow_mod(E.c, grp.get_p()) * E.a) % grp.get_p();

      return u == w;
      }
};

} // end namespace

#endif
