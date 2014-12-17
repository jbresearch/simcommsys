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

#include "map_straight.h"
#include "vectorutils.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Matrix: show input/output sizes on transform/inverse
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*** Vector Specialization ***/

// Description

template <class dbl>
std::string map_straight<libbase::vector, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Vector)";
   sout << " [" << this->input_block_size() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<libbase::vector, dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_straight<libbase::vector, dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

/*** Matrix Specialization ***/

// Description

template <class dbl>
std::string map_straight<libbase::matrix, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Matrix) ";
   sout << " [" << this->input_block_size().rows() << "×"
         << this->input_block_size().cols() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<libbase::matrix, dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_straight<libbase::matrix, dbl>::serialize(std::istream& sin)
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
   (vector)(matrix)
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_straight<container,real>
 * where:
 *      container = vector | matrix
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_straight<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_straight<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_straight<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            map_straight<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
