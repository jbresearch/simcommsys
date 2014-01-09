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

#include "sha.h"

#include <sstream>

namespace libcomm {

// Static values

bool sha::tested = false;

// Const values

const libbase::int32u sha::K[] = {0x5a827999, 0x6ed9eba1, 0x8f1bbcdc,
      0xca62c1d6};

// Construction/Destruction

sha::sha()
   {
   // reset chaining variables
   m_hash.init(5);
   m_hash = 0;
   // set byte-order flag
   lsbfirst = false;
   // perform implementation tests on algorithm, exit on failure
   if (!tested)
      selftest();
   }

// Digest-specific functions

void sha::derived_reset()
   {
   // reset chaining variables
   m_hash.init(5);
   m_hash(0) = 0x67452301;
   m_hash(1) = 0xefcdab89;
   m_hash(2) = 0x98badcfe;
   m_hash(3) = 0x10325476;
   m_hash(4) = 0xc3d2e1f0;
   }

void sha::process_block(const libbase::vector<libbase::int32u>& M)
   {
   // create expanded message block
   libbase::vector<libbase::int32u> W;
   expand(M, W);
   // copy variables
   libbase::vector<libbase::int32u> hash = m_hash;
   // main loop
   for (int t = 0; t < 80; t++)
      {
      const libbase::int32u temp = cshift(hash(0), 5) + f(t, hash(1), hash(2),
            hash(3)) + hash(4) + W(t) + K[t / 20];
      hash(4) = hash(3);
      hash(3) = hash(2);
      hash(2) = cshift(hash(1), 30);
      hash(1) = hash(0);
      hash(0) = temp;
      //trace << "SHA: step " << t << "\t" << hex << hash(0) << " " << hash(1) << " " << hash(2) << " " << hash(3) << " " << hash(4) << dec << std::endl;
      }
   // add back variables
   m_hash += hash;
   }

// Verification function

void sha::selftest()
   {
   // set flag to avoid re-entry
   tested = true;
   libbase::trace << "sha: Testing implementation" << std::endl;
   // http://www.faqs.org/rfcs/rfc3174.html
   std::string sMessage, sHash;
   // Test libbase::vector 0
   sMessage = "";
   sHash = "da39a3ee5e6b4b0d3255bfef95601890afd80709";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 1
   sMessage = "abc";
   sHash = "a9993e364706816aba3e25717850c26c9cd0d89d";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 2
   sMessage = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
   sHash = "84983e441c3bd26ebaae4aa1f95129e5e54670f1";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 3
   sMessage = "";
   for (int i = 0; i < 1000000; i++)
      sMessage += "a";
   sHash = "34aa973cd4c4daa4f61eeb2bdbad27316534016f";
   assertalways(verify(sMessage,sHash));
   // Test libbase::vector 4
   sMessage = "";
   for (int i = 0; i < 10; i++)
      sMessage
            += "0123456701234567012345670123456701234567012345670123456701234567";
   sHash = "dea356a2cddd90c7a7ecedc5ebb563934f460452";
   assertalways(verify(sMessage,sHash));
   }

bool sha::verify(const std::string message, const std::string hash)
   {
   sha d;
   d.reset();
   // process requires a pass by reference, which cannot be done by
   // direct conversion.
   std::istringstream s(message);
   d.process(s);
   return hash == std::string(d);
   }

// SHA nonlinear function implementations

libbase::int32u sha::f(const int t, const libbase::int32u X,
      const libbase::int32u Y, const libbase::int32u Z)
   {
   assert(t<80);
   switch (t / 20)
      {
      case 0:
         return (X & Y) | ((~X) & Z);
      case 1:
         return X ^ Y ^ Z;
      case 2:
         return (X & Y) | (X & Z) | (Y & Z);
      case 3:
         return X ^ Y ^ Z;
      }
   return 0;
   }

// Circular shift function

libbase::int32u sha::cshift(const libbase::int32u x, const int s)
   {
   return (x << s) | (x >> (32 - s));
   }

// Message expansion function

void sha::expand(const libbase::vector<libbase::int32u>& M, libbase::vector<
      libbase::int32u>& W)
   {
   // check input size
   assert(M.size() == 16);
   // set up output size
   W.init(80);
   // initialize values
   int i;
   for (i = 0; i < 16; i++)
      W(i) = M(i);
   for (i = 16; i < 80; i++)
      W(i) = cshift(W(i - 3) ^ W(i - 8) ^ W(i - 14) ^ W(i - 16), 1);
   }

} // end namespace
