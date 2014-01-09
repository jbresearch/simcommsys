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

#include "map_interleaved.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

/*** Vector Specialization ***/

// Interface with mapper
template <class dbl>
void map_interleaved<libbase::vector, dbl>::advance() const
   {
   lut.init(This::output_block_size(), r);
   }

template <class dbl>
void map_interleaved<libbase::vector, dbl>::dotransform(const array1i_t& in,
      array1i_t& out) const
   {
   assert(in.size() == lut.size());
   // final vector is the same size as input one
   out.init(lut.size());
   // shuffle the results
   for (int i = 0; i < lut.size(); i++)
      out(lut(i)) = in(i);
   }

template <class dbl>
void map_interleaved<libbase::vector, dbl>::dotransform(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   assert(pin.size() == lut.size());
   // final matrix is the same size as input
   pout.init(lut.size());
   // shuffle the likelihood tables
   for (int i = 0; i < lut.size(); i++)
      pout(lut(i)) = pin(i);
   }

template <class dbl>
void map_interleaved<libbase::vector, dbl>::doinverse(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   assert(pin.size() == lut.size());
   // final matrix is the same size as input
   pout.init(lut.size());
   // invert the shuffling
   for (int i = 0; i < lut.size(); i++)
      pout(i) = pin(lut(i));
   }

// Description

template <class dbl>
std::string map_interleaved<libbase::vector, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Interleaved Mapper";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_interleaved<libbase::vector, dbl>::serialize(
      std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_interleaved<libbase::vector, dbl>::serialize(
      std::istream& sin)
   {
   return sin;
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;
using libbase::matrix;
using libbase::vector;

#define CONTAINER_TYPE_SEQ \
   (vector)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_interleaved<container,real>
 * where:
 *      container = vector
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_interleaved<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_interleaved<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_interleaved<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            map_interleaved<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
