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

#include "qsc.h"
#include <sstream>

namespace libcomm {

// Channel function overrides

/*!
 * \copydoc channel::corrupt()
 *
 * The channel model implemented is described by the following state diagram:
 * \dot
 * digraph states {
 * // Make figure left-to-right
 * rankdir = LR;
 * // state definitions
 * this [ shape=circle, color=gray, style=filled, label="t(i)" ];
 * next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
 * // path definitions
 * this -> next [ label="1-Ps" ];
 * this -> Substitute [ label="Ps" ];
 * Substitute -> next;
 * }
 * \enddot
 *
 * For symbols that are substituted, any of the remaining symbols are equally
 * likely.
 */
template <class G>
G qsc<G>::corrupt(const G& s)
   {
   const double p = this->r.fval_closed();
   if (p < Ps)
      return field_utils<G>::corrupt(s, this->r);
   return s;
   }

// description output

template <class G>
std::string qsc<G>::description() const
   {
   std::ostringstream sout;
   sout << field_utils<G>::elements() << "-ary Symmetric channel";
   return sout.str();
   }

// object serialization - saving

template <class G>
std::ostream& qsc<G>::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

template <class G>
std::istream& qsc<G>::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

/* Serialization string: qsc<type>
 * where:
 *      type = bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
   template class qsc<type>; \
   template <> \
   const serializer qsc<type>::shelper( \
         "channel", \
         "qsc<" BOOST_PP_STRINGIZE(type) ">", \
         qsc<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // end namespace
