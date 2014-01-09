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

#ifndef FIELD_UTILS_H_
#define FIELD_UTILS_H_

#include "config.h"
#include "random.h"
#include "vector.h"

namespace libcomm {

/*!
 * \brief   Helper class for field types - template base
 * \author  Johann Briffa
 *
 * This class is a helper implementing various methods required when using field
 * types as symbols. The template base implements the methods as suitable for
 * the gf class.
 */

template <class G>
class field_utils {
private:
   //! Private constructor to prohibit objects of this type
   field_utils()
      {
      }
public:
   //! Field size (number of elements)
   static int elements()
      {
      return G::elements();
      }
   //! Cause a substitution error to given symbol (all other values equally likely)
   static G corrupt(const G& s, libbase::randgen& r)
      {
      return s + G(r.ival(G::elements() - 1) + 1);
      }
   //! Add vectors of symbols, modulo the field size - binary operator
   static libbase::vector<G> add(const libbase::vector<G>& a,
         const libbase::vector<G>& b)
      {
      return a + b;
      }
   //! Add vectors of symbols, modulo the field size - unary operator
   static libbase::vector<G>& add_to(libbase::vector<G>& a,
         const libbase::vector<G>& b)
      {
      return a += b;
      }
};

/*!
 * \brief   Helper class for field types - bool specialization
 * \author  Johann Briffa
 *
 * This template specialization implements the methods suitable for bool types.
 */

template <>
class field_utils<bool> {
private:
   //! Private constructor to prohibit objects of this type
   field_utils()
      {
      }
public:
   //! Field size (number of elements)
   static int elements()
      {
      return 2;
      }
   //! Cause a substitution error to given symbol
   static bool corrupt(const bool& s, libbase::randgen& r)
      {
      return !s;
      }
   //! Add vectors of symbols, modulo 2 - binary operator
   static libbase::vector<bool> add(const libbase::vector<bool>& a,
         const libbase::vector<bool>& b)
      {
      return a ^ b;
      }
   //! Add vectors of symbols, modulo 2 - unary operator
   static libbase::vector<bool>& add_to(libbase::vector<bool>& a,
         const libbase::vector<bool>& b)
      {
      return a ^= b;
      }
};

} // end namespace

#endif /* FIELD_UTILS_H_ */
