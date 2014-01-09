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

#include "zsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Internal functions

// FSM operations (advance/output/step)

template <class S>
libbase::vector<int> zsm<S>::output(const libbase::vector<int>& input) const
   {
   assert(input.size() == 1);
   // Compute output
   libbase::vector<int> op(r);
   for (int j = 0; j < r; j++)
      {
      op(j) = input(0);
      }
   return op;
   }

// Description & Serialization

//! Description output - common part only, must be preceded by specific name
template <class S>
std::string zsm<S>::description() const
   {
   std::ostringstream sout;
   sout << "Repetition code (q=" << S::elements() << ", r=" << r << ")";
   return sout.str();
   }

template <class S>
std::ostream& zsm<S>::serialize(std::ostream& sout) const
   {
   sout << "#: Repetition count" << std::endl;
   sout << r << std::endl;
   return sout;
   }

template <class S>
std::istream& zsm<S>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> r >> libbase::verify;
   return sin;
   }

} // end namespace

#include "gf.h"
#include "symbol.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::symbol;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define ALL_SYMBOL_TYPE_SEQ \
   GF_TYPE_SEQ \
   SYMBOL_TYPE_SEQ

/* Serialization string: zsm<type>
 * where:
 *      type = gf2 | gf4 ... | symbol<n> (n=2..100)
 */
#define INSTANTIATE(r, x, type) \
   template class zsm<type>; \
   template <> \
   const serializer zsm<type>::shelper( \
         "fsm", \
         "zsm<" BOOST_PP_STRINGIZE(type) ">", \
         zsm<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, ALL_SYMBOL_TYPE_SEQ)

} // end namespace
