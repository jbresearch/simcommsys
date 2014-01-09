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

#ifndef __crypt_h
#define __crypt_h

#include "config.h"
#include "vector.h"

#include <string>
#include <iostream>

namespace libcomm {

/*!
 * \brief   UNIX Crypt Algorithm.
 * \author  Johann Briffa
 *
 * \version 0.01 (28 Feb 2003)
 * initial version - class that implements UNIX Crypt Algorithm, built by
 * following the Cygwin (GNU) sources; intended for major upgrade.
 *
 * \version 0.02 (17 Jul 2006)
 * in crypt_main, made explicit conversion from char to int of array subscript.
 *
 * \version 1.00 (30 Oct 2006)
 * - defined class and associated data within "libcomm" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 */

class crypt {
   // constants - scalars
   static const int BS, BS2, KS, KS2, IS, IS2;
   // constants - vectors & matrices
   static const char PC1[];
   static const char PC2[];
   static const char IP[];
   static const char EP[];
   static const char E0[];
   static const char PERM[];
   static const char S_BOX[][64];
   // working spaces
   char E[48];
   char schluessel[16][48];
public:
   // basic constructor/destructor
   crypt();
   virtual ~crypt();
   // public functions
   void encrypt(char *nachr, int decr);
   void setkey(char *schl);
   char *crypttext(const char *wort, const char *salt);
protected:
   // private functions
   static void perm(char *a, const char *e, const char *pc, int n);
   void crypt_main(char *nachr_l, char *nachr_r, char *schl);
};

} // end namespace

#endif
