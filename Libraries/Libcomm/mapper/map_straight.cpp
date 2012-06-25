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
 * 
 * \section svn Version Control
 * - $Id$
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

using libbase::vector;

// Interface with mapper

/*! \copydoc mapper::setup()
 * 
 * \note Each encoder output must be represented by an integral number of
 * modulation symbols
 */
template <class dbl>
void map_straight<vector, dbl>::setup()
   {
   s1 = get_rate(Base::M, Base::N);
   s2 = get_rate(Base::M, Base::S);
   upsilon = Base::size.length() * s1 / s2;
   assertalways(Base::size.length()*s1 == upsilon*s2);
   }

template <class dbl>
void map_straight<vector, dbl>::dotransform(const array1i_t& in, array1i_t& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results vector
   out.init(This::output_block_size());
   // Modulate encoded stream (least-significant first)
   for (int t = 0, k = 0; t < Base::size.length(); t++)
      for (int i = 0, x = in(t); i < s1; i++, k++, x /= Base::M)
         out(k) = x % Base::M;
   }

template <class dbl>
void map_straight<vector, dbl>::doinverse(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == Base::M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   libbase::allocate(pout, upsilon, Base::S);
   // Get the necessary data from the channel
   for (int t = 0; t < upsilon; t++)
      for (int x = 0; x < Base::S; x++)
         {
         pout(t)(x) = 1;
         for (int i = 0, thisx = x; i < s2; i++, thisx /= Base::M)
            pout(t)(x) *= pin(t * s2 + i)(thisx % Base::M);
         }
   }

// Description

template <class dbl>
std::string map_straight<vector, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Vector)";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<vector, dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_straight<vector, dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

/*** Matrix Specialization ***/

using libbase::matrix;

// Interface with mapper

/*! \copydoc mapper::setup()
 * 
 * \note Symbol alphabets must be the same size
 */
template <class dbl>
void map_straight<matrix, dbl>::setup()
   {
   assertalways(Base::M == Base::N);
   assertalways(Base::M == Base::S);
   }

template <class dbl>
void map_straight<matrix, dbl>::dotransform(const array2i_t& in, array2i_t& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results matrix
   out.init(This::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_straight): Transform ";
   libbase::trace << in.size().rows() << "x" << in.size().cols() << " to ";
   libbase::trace << out.size().rows() << "x" << out.size().cols() << std::endl;
#endif
   // Map encoded stream (row-major order)
   int ii = 0, jj = 0;
   for (int i = 0; i < in.size().rows(); i++)
      for (int j = 0; j < in.size().cols(); j++)
         {
         out(ii, jj) = in(i, j);
         jj++;
         if (jj >= out.size().cols())
            {
            jj = 0;
            ii++;
            }
         }
   }

template <class dbl>
void map_straight<matrix, dbl>::doinverse(const array2vd_t& pin,
      array2vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0,0).size() == Base::M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   pout.init(This::input_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_straight): Inverse ";
   libbase::trace << pin.size().rows() << "x" << pin.size().cols() << " to ";
   libbase::trace << pout.size().rows() << "x" << pout.size().cols() << std::endl;
#endif
   // Map channel receiver information (row-major order)
   int ii = 0, jj = 0;
   for (int i = 0; i < pin.size().rows(); i++)
      for (int j = 0; j < pin.size().cols(); j++)
         {
         pout(ii, jj) = pin(i, j);
         jj++;
         if (jj >= pout.size().cols())
            {
            jj = 0;
            ii++;
            }
         }
   }

// Description

template <class dbl>
std::string map_straight<matrix, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Matrix ";
   sout << size_out.rows() << "x" << size_out.cols() << ")";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<matrix, dbl>::serialize(std::ostream& sout) const
   {
   sout << size_out << std::endl;
   return sout;
   }

template <class dbl>
std::istream& map_straight<matrix, dbl>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> size_out >> libbase::verify;
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
