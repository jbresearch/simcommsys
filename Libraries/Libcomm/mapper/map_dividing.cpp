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

#include "map_dividing.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*** Vector Specialization ***/

// Interface with mapper
template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::dotransform(const array1i_t& in,
      array1i_t& out) const
   {
   // Confirm input sequence to be of the correct length
   assertalways(in.size() == this->input_block_size());
   // Create converter object and perform necessary transform
   libbase::symbol_converter<dbl,dbl2> converter(Base::M, Base::q);
   converter.divide_symbols(in, out);
   }

template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::dotransform(
      const array1vd_t& pin, array1vd_t& pout) const
   {
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == this->input_block_size());
   // Create converter object and perform necessary transform
   libbase::symbol_converter<dbl,dbl2> converter(Base::M, Base::q);
   converter.divide_probabilities(pin, pout);
   }

template <class dbl, class dbl2>
void map_dividing<libbase::vector, dbl, dbl2>::doinverse(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Create converter object and perform necessary transform
   libbase::symbol_converter<dbl,dbl2> converter(Base::M, Base::q);
   converter.aggregate_probabilities(pin, pout);
   }

// Description

template <class dbl, class dbl2>
std::string map_dividing<libbase::vector, dbl, dbl2>::description() const
   {
   std::ostringstream sout;
   sout << "Dividing Mapper (Vector)";
   sout << " [" << this->input_block_size() << "<->"
         << this->output_block_size() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl, class dbl2>
std::ostream& map_dividing<libbase::vector, dbl, dbl2>::serialize(
      std::ostream& sout) const
   {
   return sout;
   }

template <class dbl, class dbl2>
std::istream& map_dividing<libbase::vector, dbl, dbl2>::serialize(
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
using libbase::vector;

#define CONTAINER_TYPE_SEQ \
   (vector)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_dividing<container,dbl,dbl2>
 * where:
 *      container = vector
 *      dbl = float | double | logrealfast
 *              [dbl is the interface arithmetic type]
 *      dbl2 = float | double | logrealfast
 *              [dbl2 is the internal arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_dividing<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_dividing<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_dividing<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            map_dividing<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
