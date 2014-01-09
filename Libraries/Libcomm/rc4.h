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

#ifndef __rc4_h
#define __rc4_h

#include "config.h"
#include "vector.h"

#include <string>

namespace libcomm {

/*!
 * \brief   RSA RC4 Algorithm.
 * \author  Johann Briffa
 *
 * \version 1.00 (03 Jul 2003)
 * initial version - class that implements RSA RC4 Algorithm, as specified in
 * Schneier, "Applied Cryptography", 1996, pp.397-398.
 *
 * \version 1.01 (04 Jul 2003)
 * changed vector tables to int8u instead of int, to ensure validity of values.
 *
 * \version 1.02 (5 Jul 2003)
 * - added self-testing on creation of the first object.
 * - modified counters to be int8u instead of int - also renamed them x & y
 * - removed superfluous mod 256 (& 0xff) operations
 *
 * \version 1.03 (17 Jul 2006)
 * in encrypt, changed the loop variable to type size_t, to avoid the warning about
 * comparisons between signed and unsigned types.
 *
 * \version 1.10 (6 Nov 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class rc4 {
   // static variables
   static bool tested;
   // working spaces
   libbase::vector<libbase::int8u> S;
   libbase::int8u x, y;
public:
   // basic constructor/destructor
   rc4();
   virtual ~rc4();
   // public functions
   void init(std::string key);
   std::string encrypt(const std::string plaintext);
   libbase::int8u encrypt(const libbase::int8u plaintext);
protected:
   // private functions
   bool verify(const std::string key, const std::string plaintext,
         const std::string ciphertext);
};

} // end namespace

#endif
