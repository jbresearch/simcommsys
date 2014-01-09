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

#include "laplacian.h"

namespace libcomm {

// *** general template ***

// object serialization

template <class S, template <class > class C>
std::ostream& laplacian<S, C>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class S, template <class > class C>
std::istream& laplacian<S, C>::serialize(std::istream& sin)
   {
   return sin;
   }

// *** sigspace partial specialization ***

// object serialization

template <template <class > class C>
std::ostream& laplacian<sigspace, C>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <template <class > class C>
std::istream& laplacian<sigspace, C>::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::serializer;
using libbase::matrix;
using libbase::vector;

#define SYMBOL_TYPE_SEQ \
   (int)(float)(double)(sigspace)
#define CONTAINER_TYPE_SEQ \
   (vector)(matrix)

/* Serialization string: laplacian<type,container>
 * where:
 *      type = int | float | double | sigspace
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class laplacian<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer laplacian<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "channel", \
            "laplacian<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            laplacian<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
