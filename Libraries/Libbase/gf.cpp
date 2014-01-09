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

#include "gf.h"

#include <cstdlib>
#include <string>

namespace libbase {

using std::cerr;

// Internal functions

/*!
 * \brief Conversion from string
 * \param   s     String representation of element by its polynomial coefficients (binary)
 *
 * This function converts the string to an integer and calls init().
 * The string must only contain 1's and 0's.
 */
template <int m, int poly>
void gf<m, poly>::init(const char *s)
   {
   int32u value = 0;
   const char *p;
   for (p = s; *p == '1' || *p == '0'; p++)
      {
      value <<= 1;
      if (*p == '1')
         value |= 1;
      }
   assert(*p == '\0');
   init(value);
   }

// Conversion operations

template <int m, int poly>
gf<m, poly>::operator std::string() const
   {
   std::string sTemp;
   for (int i = m - 1; i >= 0; i--)
      sTemp += '0' + ((value >> i) & 1);
   return sTemp;
   }

// Explicit Realizations

// Degenerate case GF(2)

template class gf<1, 0x3> ; // 1 { 1 }

// cf. Lin & Costello, 2004, App. A

template class gf<2, 0x7> ; // 1 { 11 }
template class gf<3, 0xB> ; // 1 { 011 }
template class gf<4, 0x13> ; // 1 { 0011 }
template class gf<5, 0x25> ; // 1 { 0 0101 }
template class gf<6, 0x43> ; // 1 { 00 0011 }
template class gf<7, 0x89> ; // 1 { 000 1001 }
template class gf<8, 0x11D> ; // 1 { 0001 1101 }
template class gf<9, 0x211> ; // 1 { 0 0001 0001 }
template class gf<10, 0x409> ; // 1 { 00 0000 1001 }

// Rijndael field cf. Gladman, 2003, p.5

template class gf<8, 0x11B> ; // 1 { 0001 1011 }

/* NOTE: We cannot use the following loop because C++ does not allow explicit
 * instantiation using typedefs...
   #include <boost/preprocessor/seq/for_each.hpp>

   #define INSTANTIATE(r, x, type) \
         template class type;

   BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, GF_TYPE_SEQ)
 */

} // end namespace
