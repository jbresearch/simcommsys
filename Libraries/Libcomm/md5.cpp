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

#include "md5.h"

#include <cmath>
#include <sstream>

namespace libcomm {

// Static values

bool md5::tested = false;
libbase::vector<libbase::int32u> md5::t;

// Const values

const int md5::s[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17,
      22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16,
      23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10,
      15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

const int md5::ndx[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 5, 8, 11, 14, 1, 4,
      7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6,
      13, 4, 11, 2, 9};

// Construction/Destruction

md5::md5()
   {
   // reset chaining variables
   m_hash.init(4);
   m_hash = 0;
   // set byte-order flag
   lsbfirst = true;
   // initialise constants if not yet done
   if (t.size() == 0)
      {
      t.init(64);
      for (int i = 0; i < 64; i++)
         t(i) = libbase::int32u(floor(pow(double(2), 32) * fabs(sin(double(i
               + 1)))));
      }
   // perform implementation tests on algorithm, exit on failure
   if (!tested)
      selftest();
   }

// Public interface for computing digest

void md5::derived_reset()
   {
   // reset chaining variables
   m_hash.init(4);
   m_hash(0) = 0x67452301;
   m_hash(1) = 0xefcdab89;
   m_hash(2) = 0x98badcfe;
   m_hash(3) = 0x10325476;
   }

void md5::process_block(const libbase::vector<libbase::int32u>& M)
   {
   // copy variables
   libbase::vector<libbase::int32u> hash = m_hash;
   // main loop
   for (int i = 0; i < 64; i++)
      {
      int a = (64 - i) & 0x3;
      int b = (65 - i) & 0x3;
      int c = (66 - i) & 0x3;
      int d = (67 - i) & 0x3;
      hash(a) = op(i, hash(a), hash(b), hash(c), hash(d), M);
      }
   // add back variables
   m_hash += hash;
   }

// Verification function

void md5::selftest()
   {
   // set flag to avoid re-entry
   tested = true;
   libbase::trace << "md5: Testing implementation" << std::endl;
   // http://www.faqs.org/rfcs/rfc1321.html
   std::string sMessage, sHash;
   // Test libbase::vector 0
   sMessage = "";
   sHash = "d41d8cd98f00b204e9800998ecf8427e";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 1
   sMessage = "a";
   sHash = "0cc175b9c0f1b6a831c399e269772661";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 2
   sMessage = "abc";
   sHash = "900150983cd24fb0d6963f7d28e17f72";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 3
   sMessage = "message digest";
   sHash = "f96b697d7cb7938d525a2f31aaf161d0";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 4
   sMessage = "abcdefghijklmnopqrstuvwxyz";
   sHash = "c3fcd3d76192e4007dfb496cca67e13b";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 5
   sMessage = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
   sHash = "d174ab98d277d9f5a5611c2c9f419d9f";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 6
   sMessage
         = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
   sHash = "57edf4a22be3c955ac49da2e2107b67a";
   assertalways(verify(sMessage,sHash));
   }

bool md5::verify(const std::string message, const std::string hash)
   {
   md5 d;
   d.reset();
   // process requires a pass by reference, which cannot be done by
   // direct conversion.
   std::istringstream s(message);
   d.process(s);
   return hash == std::string(d);
   }

// Circular shift function

libbase::int32u md5::cshift(const libbase::int32u x, const int s)
   {
   return (x << s) | (x >> (32 - s));
   }

// MD5 nonlinear function implementations

libbase::int32u md5::f(const int i, const libbase::int32u X,
      const libbase::int32u Y, const libbase::int32u Z)
   {
   assert(i<64);
   switch (i / 16)
      {
      case 0:
         return (X & Y) | ((~X) & Z);
      case 1:
         return (X & Z) | (Y & (~Z));
      case 2:
         return X ^ Y ^ Z;
      case 3:
         return Y ^ (X | (~Z));
      }
   return 0;
   }

// Circular shift function

libbase::int32u md5::op(const int i, const libbase::int32u a,
      const libbase::int32u b, const libbase::int32u c,
      const libbase::int32u d, const libbase::vector<libbase::int32u>& M)
   {
   return b + cshift(a + f(i, b, c, d) + M(ndx[i]) + t(i), s[i]);
   }

} // end namespace
