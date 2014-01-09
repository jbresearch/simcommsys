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

#include "named_lut.h"
#include <sstream>

namespace libcomm {

// description output

template <class real>
std::string named_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Named Interleaver (" << lutname;
   if (m > 0)
      sout << ", Forced tail length " << m << ")";
   else
      sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& named_lut<real>::serialize(std::ostream& sout) const
   {
   sout << m << std::endl;
   sout << lutname << std::endl;
   sout << this->lut;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& named_lut<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> m >> libbase::verify;
   sin >> libbase::eatcomments >> lutname >> libbase::verify;
   sin >> libbase::eatcomments >> this->lut >> libbase::verify;
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: named_lut<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class named_lut<type>; \
   template <> \
   const serializer named_lut<type>::shelper( \
         "interleaver", \
         "named_lut<" BOOST_PP_STRINGIZE(type) ">", \
         named_lut<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
