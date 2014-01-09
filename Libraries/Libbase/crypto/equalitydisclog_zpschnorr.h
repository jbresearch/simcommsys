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

#ifndef __equalitydisclog_zpschnorr_h
#define __equalitydisclog_zpschnorr_h

#include "config.h"
#include "group.h"
#include "keygenerator.h"
#include "sha.h"

namespace libbase {

/*!
 * \brief   Log-domain comparison
 * \author  Johann Briffa
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class equalitydisclog_zpschnorr {
private:
   BigInteger g1;
   BigInteger g2;

   BigInteger v;
   BigInteger w;

   BigInteger a;
   BigInteger b;
   BigInteger c;
   BigInteger r;

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   explicit equalitydisclog_zpschnorr(BigInteger g1, BigInteger g2,
         BigInteger v, BigInteger w, BigInteger a, BigInteger b, BigInteger c,
         BigInteger r) :
      g1(g1), g2(g2), v(v), w(w), a(a), b(b), c(c), r(r)
      {
      }
   // @}

   /* \brief Proof construction
    *
    * To prove that log v = log w, where v = g_1^x and w = g_2^x, let:
    *      z = random in Z_q
    *      a = g_1^z
    *      b = g_2^z
    *      c = hash(v,w,a,b)
    *      r = (z + cx) mod q
    *
    * The proof is (a,b,c,r).
    */
   static equalitydisclog_zpschnorr constructProof(group<BigInteger> grp,
         BigInteger g1, BigInteger g2, BigInteger x)
      {
      BigInteger v = g1.pow_mod(x, grp.get_p());
      BigInteger w = g2.pow_mod(x, grp.get_p());
      BigInteger z = group<BigInteger>::get_random_integer(grp.get_q());

      BigInteger a = g1.pow_mod(z, grp.get_p());
      BigInteger b = g2.pow_mod(z, grp.get_p());

      std::vector<unsigned char> buf;
      buf += v.bytearray();
      buf += w.bytearray();
      buf += a.bytearray();
      buf += b.bytearray();

      libcomm::sha md;
      md.process(buf);

      BigInteger c(md);
      c %= grp.get_q();

      BigInteger r = (z + (c * x)) % grp.get_q();

      return equalitydisclog_zpschnorr(g1, g2, v, w, a, b, c, r);
      }

   /* \brief Proof verification
    *
    * To prove that log v = log w, where v = g_1^x and w = g_2^x, let:
    *      z = random in Z_q
    *      a = g_1^z
    *      b = g_2^z
    *      c = hash(v,w,a,b)
    *      r = (z + cx) mod q
    *
    * To verify, check that g_1^r = av^c (mod p) and g_2^r = bw^c (mod p).
    */
   static bool verify(group<BigInteger> grp, equalitydisclog_zpschnorr E)
      {
      const BigInteger lhs1 = E.g1.pow_mod(E.r, grp.get_p());
      const BigInteger lhs2 = E.g2.pow_mod(E.r, grp.get_p());
      const BigInteger rhs1 = (E.v.pow_mod(E.c, grp.get_p()) * E.a)
            % grp.get_p();
      const BigInteger rhs2 = (E.w.pow_mod(E.c, grp.get_p()) * E.b)
            % grp.get_p();

      return lhs1 == rhs1 && lhs2 == rhs2;
      }
};

} // end namespace

#endif
