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

#include "qec.h"
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
 * this -> next [ label="1-Pe" ];
 * this -> Erase [ label="Pe" ];
 * Erase -> next;
 * }
 * \enddot
 */
template <class G>
G qec<G>::corrupt(const G& s)
   {
   const double p = this->r.fval_closed();
   G rx = s;
   if (p < Pe)
      rx.erase();
   return rx;
   }

// description output

template <class G>
std::string qec<G>::description() const
   {
   std::ostringstream sout;
   sout << G::elements() << "-ary Erasure channel";
   return sout.str();
   }

// object serialization - saving

template <class G>
std::ostream& qec<G>::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

template <class G>
std::istream& qec<G>::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace

#include "gf.h"
#include "erasable.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::erasable;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

#define SET_ERASABLE(r, x, type) \
   (erasable<type>)

#define ALL_SYMBOL_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(SET_ERASABLE, x, SYMBOL_TYPE_SEQ)


/* Serialization string: qec<type>
 * where:
 *      type = erasable<bool | gf2 | gf4 ...>
 */
#define INSTANTIATE(r, x, type) \
   template class qec<type>; \
   template <> \
   const serializer qec<type>::shelper( \
         "channel", \
         "qec<" BOOST_PP_STRINGIZE(type) ">", \
         qec<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, ALL_SYMBOL_TYPE_SEQ)

} // end namespace
