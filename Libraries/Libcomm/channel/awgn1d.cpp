/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
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

#include "awgn1d.h"

namespace libcomm
{

// Serialization Support

template <class S>
std::ostream&
awgn1d<S>::serialize(std::ostream& sout) const
{
    return sout;
}

template <class S>
std::istream&
awgn1d<S>::serialize(std::istream& sin)
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
#define REAL_TYPE_SEQ \
   (float)(double)

/* Serialization string: awgn1d<real>
 * where:
 *      real = float | double
 */
#define INSTANTIATE(r, x, type) \
      template class awgn1d<type>; \
      template <> \
      const serializer awgn1d<type>::shelper( \
            "channel", \
            "awgn1d<" BOOST_PP_STRINGIZE(type) ">", \
            awgn1d<type>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // namespace libcomm
