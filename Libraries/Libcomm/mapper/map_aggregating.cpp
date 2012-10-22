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

#include "map_aggregating.h"
#include "vectorutils.h"
#include <cstdlib>
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

template <class dbl>
void map_aggregating<libbase::vector, dbl>::dotransform(const array1i_t& in,
      array1i_t& out) const
   {
   // Confirm input sequence to be of the correct length
   assertalways(in.size() == This::input_block_size());
   // Initialize results vector
   out.init(This::output_block_size());
   // Aggregate encoded symbols into modulation symbols (least-significant first)
   for (int t = 0, k = 0; t < out.size(); t++)
      {
      int temp = 0;
      for (int i = 0, mul = 1; i < n_per_m; i++, k++, mul *= Base::N)
         {
         assert(in(k) >= 0 && in(k) < Base::N);
         temp += in(k) * mul;
         }
      assert(temp >= 0 && temp < Base::M);
      out(t) = temp;
      }
   }

template <class dbl>
void map_aggregating<libbase::vector, dbl>::dotransform(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm input symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == Base::N);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::input_block_size());
   // Initialize results vector
   libbase::allocate(pout, This::output_block_size(), Base::M);
   // Transform the encoder output posteriors to blockmodem priors
   for (int t = 0; t < This::output_block_size(); t++)
      for (int x = 0; x < Base::M; x++)
         {
         double temp = 1;
         for (int i = 0, thisx = x; i < n_per_m; i++, thisx /= Base::N)
            temp *= pin(t * n_per_m + i)(thisx % Base::N);
         pout(t)(x) = temp;
         }
   }

template <class dbl>
void map_aggregating<libbase::vector, dbl>::doinverse(const array1vd_t& pin,
      array1vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == Base::M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   libbase::allocate(pout, This::input_block_size(), Base::N);
   // Get the necessary data from the channel
   for (int t = 0, mul = 1; t < This::input_block_size(); t++, mul *= Base::N)
      {
      const int k = t / n_per_m;
      const int offset = t - k * n_per_m;
      if (offset == 0)
         mul = 1;
      for (int x = 0; x < Base::N; x++)
         {
         double temp = 0;
         for (int d = 0; d < Base::M; d++)
            if ((d / mul) % Base::N == x)
               temp += pin(k)(d);
         pout(t)(x) = temp;
         }
      }
   }

// Description

template <class dbl>
std::string map_aggregating<libbase::vector, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Aggregating Mapper (Vector)";
   sout << " [" << this->input_block_size() << "<->"
         << this->output_block_size() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_aggregating<libbase::vector, dbl>::serialize(
      std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_aggregating<libbase::vector, dbl>::serialize(
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

/* Serialization string: map_aggregating<container,real>
 * where:
 *      container = vector
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class map_aggregating<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer map_aggregating<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "mapper", \
            "map_aggregating<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            map_aggregating<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
