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

#include "flat.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// initialization

template <class real>
void flat<real>::init(const int tau)
   {
   this->lut.init(tau);
   for (int i = 0; i < tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string flat<real>::description() const
   {
   std::ostringstream sout;
   sout << "Flat Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& flat<real>::serialize(std::ostream& sout) const
   {
   sout << "# Interleaver size" << std::endl;
   sout << this->lut.size() << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& flat<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   init(tau);
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

/* Serialization string: flat<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
   template class flat<type>; \
   template <> \
   const serializer flat<type>::shelper( \
         "interleaver", \
         "flat<" BOOST_PP_STRINGIZE(type) ">", \
         flat<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
