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

#include "map_stipple.h"
#include "vectorutils.h"
#include <cstdlib>
#include <sstream>

namespace libcomm
{

/*** Vector Specialization ***/

// Interface with mapper
template <class dbl>
void map_stipple<libbase::vector, dbl>::advance() const
{
    assertalways(Base::size > 0);
    assertalways(sets > 0);

    // check if matrix is already set
    if (pattern.size() == Base::size.length()) {
        return;
    }

    // compute required sizes
    const int tau = Base::size.length() / (sets + 1);
    assertalways(Base::size.length() == tau * (sets + 1));

    // initialise the pattern matrix
    pattern.init(Base::size.length());
    for (int i = 0, t = 0; t < tau; t++) {
        for (int s = 0; s <= sets; s++, i++) {
            pattern(i) = (s == 0 || (s - 1) == t % sets);
        }
    }
}

template <class dbl>
void map_stipple<libbase::vector, dbl>::dotransform(const array1i_t& in,
                                                    array1i_t& out) const
{
    // final vector size depends on the number of set positions
    assertalways(in.size() == pattern.size());
    out.init(This::output_block_size());

    // puncture the results
    for (int i = 0, ii = 0; i < in.size(); i++) {
        if (pattern(i)) {
            out(ii++) = in(i);
        }
    }
}

template <class dbl>
void map_stipple<libbase::vector, dbl>::dotransform(const array1vd_t& pin,
                                                    array1vd_t& pout) const
{
    assertalways(pin.size() == pattern.size());
    assertalways(pin(0).size() == Base::q);

    // final matrix size depends on the number of set positions
    libbase::allocate(pout, This::output_block_size(), Base::q);

    // puncture the likelihood tables
    for (int i = 0, ii = 0; i < pin.size(); i++) {
        if (pattern(i)) {
            pout(ii++) = pin(i);
        }
    }
}

template <class dbl>
void map_stipple<libbase::vector, dbl>::doinverse(const array1vd_t& pin,
                                                  array1vd_t& pout) const
{
    assertalways(pin.size() == This::output_block_size());
    assertalways(pin(0).size() == Base::M);

    // final matrix size depends on the number of set positions
    libbase::allocate(pout, pattern.size(), Base::M);

    // invert the puncturing
    for (int i = 0, ii = 0; i < pattern.size(); i++) {
        if (pattern(i)) {
            pout(i) = pin(ii++);
        } else {
            pout(i) = dbl(1.0 / Base::M);
        }
    }
}

// Description

template <class dbl>
std::string map_stipple<libbase::vector, dbl>::description() const
{
    std::ostringstream sout;
    sout << "Stipple Mapper (stride=" << sets << ")";
    return sout.str();
}

// Serialization Support

template <class dbl>
std::ostream&
map_stipple<libbase::vector, dbl>::serialize(std::ostream& sout) const
{
    sout << "# Stride for stipple mapper" << std::endl;
    sout << sets << std::endl;
    return sout;
}

template <class dbl>
std::istream& map_stipple<libbase::vector, dbl>::serialize(std::istream& sin)
{
    sin >> libbase::eatcomments >> sets >> libbase::verify;
    return sin;
}

} // namespace libcomm

#include "logrealfast.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::logrealfast;
using libbase::matrix;
using libbase::serializer;
using libbase::vector;

// clang-format off
#define CONTAINER_TYPE_SEQ \
   (vector)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_stipple<container,real>
 * where:
 *      container = vector
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_stipple<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_stipple<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_stipple<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            map_stipple<BOOST_PP_SEQ_ENUM(args)>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // namespace libcomm
