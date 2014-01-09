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

#include "qim.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

using libbase::serializer;

// description output

template <class S>
std::string qim<S>::description() const
   {
   std::ostringstream sout;
   if (alpha < 1.0)
      sout << "DC-";
   sout << "QIM embedder (M=" << M << ", delta=" << delta;
   if (alpha < 1.0)
      sout << ", alpha=" << alpha;
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class S>
std::ostream& qim<S>::serialize(std::ostream& sout) const
   {
   sout << "# M" << std::endl;
   sout << M << std::endl;
   sout << "# delta" << std::endl;
   sout << delta << std::endl;
   sout << "# alpha" << std::endl;
   sout << alpha << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 */

template <class S>
std::istream& qim<S>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> M >> libbase::verify;
   sin >> libbase::eatcomments >> delta >> libbase::verify;
   sin >> libbase::eatcomments >> alpha >> libbase::verify;
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define SYMBOL_TYPE_SEQ \
   (int)(float)(double)

/* Serialization string: qim<type>
 * where:
 *      type = int | float | double
 */
#define INSTANTIATE(r, x, type) \
      template class qim<type>; \
      template <> \
      const serializer qim<type>::shelper( \
            "embedder", \
            "qim<" BOOST_PP_STRINGIZE(type) ">", \
            qim<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // end namespace
