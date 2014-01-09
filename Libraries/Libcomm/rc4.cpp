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

#include "rc4.h"

#include "itfunc.h"

namespace libcomm {

using libbase::int8u;
using libbase::vector;

//////////////////////////////////////////////////////////////////////
// static values
//////////////////////////////////////////////////////////////////////

bool rc4::tested = false;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

rc4::rc4()
   {
   S.init(256);

   // perform implementation tests on algorithm, exit on failure
   if (!tested)
      {
      using libbase::dehexify;

      libbase::trace << "rc4: Testing implementation" << std::endl;
      // Date: Tue, 13 Sep 94 18:37:56 PDT
      // From: ekr@eit.COM (Eric Rescorla)
      // Message-Id: <9409140137.AA17743@eitech.eit.com>
      // Subject: RC4 compatibility testing
      // Cc: cypherpunks@toad.com
      std::string sKey, sInput, sOutput;
      // Test vector 0
      sKey = dehexify("0123456789abcdef");
      sInput = dehexify("0123456789abcdef");
      sOutput = dehexify("75b7878099e0c596");
      assert(verify(sKey,sInput,sOutput));
      assert(verify(sKey,sOutput,sInput));
      // Test vector 1
      sKey = dehexify("0123456789abcdef");
      sInput = dehexify("0000000000000000");
      sOutput = dehexify("7494c2e7104b0879");
      assert(verify(sKey,sInput,sOutput));
      assert(verify(sKey,sOutput,sInput));
      // Test vector 2
      sKey = dehexify("0000000000000000");
      sInput = dehexify("0000000000000000");
      sOutput = dehexify("de188941a3375d3a");
      assert(verify(sKey,sInput,sOutput));
      assert(verify(sKey,sOutput,sInput));
      // Test vector 3
      sKey = dehexify("ef012345");
      sInput = dehexify("00000000000000000000");
      sOutput = dehexify("d6a141a7ec3c38dfbd61");
      assert(verify(sKey,sInput,sOutput));
      assert(verify(sKey,sOutput,sInput));
      // Test vector 4
      sKey = dehexify("0123456789abcdef");
      sInput
            = dehexify(
                  "0101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101");
      sOutput
            = dehexify(
                  "7595c3e6114a09780c4ad452338e1ffd9a1be9498f813d76533449b6778dcad8c78a8d2ba9ac66085d0e53d59c26c2d1c490c1ebbe0ce66d1b6b1b13b6b919b847c25a91447a95e75e4ef16779cde8bf0a95850e32af9689444fd377108f98fdcbd4e726567500990bcc7e0ca3c4aaa304a387d20f3b8fbbcd42a1bd311d7a4303dda5ab078896ae80c18b0af66dff319616eb784e495ad2ce90d7f772a81747b65f62093b1e0db9e5ba532fafec47508323e671327df9444432cb7367cec82f5d44c0d00b67d650a075cd4b70dedd77eb9b10231b6b5b741347396d62897421d43df9b42e446e358e9c11a9b2184ecbef0cd8e7a877ef968f1390ec9b3d35a5585cb009290e2fcde7b5ec66d9084be44055a619d9dd7fc3166f9487f7cb272912426445998514c15d53a18c864ce3a2b7555793988126520eacf2e3066e230c91bee4dd5304f5fd0405b35bd99c73135d3d9bc335ee049ef69b3867bf2d7bd1eaa595d8bfc0066ff8d31509eb0c6caa006c807a623ef84c3d33c195d23ee320c40de0558157c822d4b8c569d849aed59d4e0fd7f379586b4b7ff684ed6a189f7486d49b9c4bad9ba24b96abf924372c8a8fffb10d55354900a77a3db5f205e1b99fcd8660863a159ad4abe40fa48934163ddde542a6585540fd683cbfd8c00f12129a284deacc4cdefe58be7137541c047126c8d49e2755ab181ab7e940b0c0");
      assert(verify(sKey,sInput,sOutput));
      assert(verify(sKey,sOutput,sInput));
      // return ok
      tested = true;
      }
   }

rc4::~rc4()
   {
   }

//////////////////////////////////////////////////////////////////////
// Private functions
//////////////////////////////////////////////////////////////////////

bool rc4::verify(const std::string key, const std::string plaintext,
      const std::string ciphertext)
   {
   init(key);
   return ciphertext == encrypt(plaintext);
   }

//////////////////////////////////////////////////////////////////////
// Public functions
//////////////////////////////////////////////////////////////////////

void rc4::init(std::string key)
   {
   int i;
   // sanity checks
   assert(S.size() == 256);
   libbase::trace << "RC4 initialising state from key (" << key
         << "), length = " << key.length() << std::endl;
   // initialise linearly
   for (i = 0; i < 256; i++)
      S(i) = i;
   // initialise key by repeating given sequence
   vector<int8u> K(256);
   for (i = 0; i < 256; i++)
      K(i) = key.at(i % key.length());
   // randomly permute S-box
   for (x = y = 0, i = 0; i < 256; i++, x++)
      {
      y += S(x) + K(x);
      std::swap(S(x), S(y));
      }
   // initialise counters
   x = y = 0;
   }

libbase::int8u rc4::encrypt(const libbase::int8u plaintext)
   {
   // sanity checks
   assert(S.size() == 256);
   // main algorithm
   x++;
   y += S(x);
   std::swap(S(x), S(y));
   libbase::int8u t = S(x) + S(y);
   // return result;
   return plaintext ^ S(t);
   }

std::string rc4::encrypt(const std::string plaintext)
   {
   std::string ciphertext;
   for (size_t i = 0; i < plaintext.length(); i++)
      ciphertext += encrypt(plaintext[i]);
   return ciphertext;
   }

} // end namespace
