/*!
 * \file
 *
 * Copyright (c) 2025 Johann A. Briffa
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

#include "sign.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm
{

// object serialization - saving

template <class S>
std::ostream&
sign<S>::serialize(std::ostream& sout) const
{
    return sout;
}

// object serialization - loading

template <class S>
std::istream&
sign<S>::serialize(std::istream& sin)
{
    return sin;
}

} // namespace libcomm

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

// clang-format off
#define SYMBOL_TYPE_SEQ \
   (float)(double)

/* Serialization string: sign<type>
 * where:
 *      type = int
 */
#define INSTANTIATE(r, x, type) \
      template class sign<type>; \
      template <> \
      const serializer sign<type>::shelper( \
            "informed_embedder", \
            "sign<" BOOST_PP_STRINGIZE(type) ">", \
            sign<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // namespace libcomm
