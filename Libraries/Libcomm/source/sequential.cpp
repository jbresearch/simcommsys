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

#include "sequential.h"
#include <sstream>

namespace libcomm
{

// object serialization - saving

template <class S, template <class> class C>
std::ostream& sequential<S, C>::serialize(std::ostream& sout) const
{
    sout << "# Version" << std::endl;
    sout << 1 << std::endl;
    sout << "#: input symbols - count" << std::endl;
    sout << input_vectors.size() << std::endl;
    sout << "#: input symbols - values" << std::endl;
    input_vectors.serialize(sout, '\n');
    return sout;
}

// object serialization - loading

template <class S, template <class> class C>
std::istream& sequential<S, C>::serialize(std::istream& sin)
{
    assertalways(sin.good());
    // get format version
    int version;
    sin >> libbase::eatcomments >> version;
    // read count of input symbols
    int temp;
    sin >> libbase::eatcomments >> temp >> libbase::verify;
    // read input symbols from stream
    input_vectors.init(temp);
    sin >> libbase::eatcomments;
    input_vectors.serialize(sin);
    libbase::verify(sin);
    // reset index
    index = 0;
    return sin;
}

} // namespace libcomm

#include "gf.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::matrix;
using libbase::serializer;
using libbase::vector;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (int) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
   //(vector)(matrix)

/* Serialization string: sequential<type,container>
 * where:
 *      type = int | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class sequential<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer sequential<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "source", \
            "sequential<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            sequential<BOOST_PP_SEQ_ENUM(args)>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
                              (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // namespace libcomm
