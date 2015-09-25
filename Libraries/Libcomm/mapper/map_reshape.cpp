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

#include "map_reshape.h"
#include "vectorutils.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show input/output sizes on transform/inverse
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Interface with mapper
template <class dbl>
void map_reshape<dbl>::dotransform(const array2i_t& in,
      array2i_t& out) const
   {
   assertalways(in.size() == this->input_block_size());
   // Initialize results matrix
   out.init(This::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_reshape): Transform ";
   libbase::trace << in.size().rows() << "×" << in.size().cols() << " to ";
   libbase::trace << out.size().rows() << "×" << out.size().cols() << std::endl;
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
void map_reshape<dbl>::dotransform(const array2vd_t& pin,
      array2vd_t& pout) const
   {
   // Confirm input symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0, 0).size() == Base::q);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == this->input_block_size());
   // Initialize results vector
   pout.init(This::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_reshape): Transform ";
   libbase::trace << pin.size().rows() << "×" << pin.size().cols() << " to ";
   libbase::trace << pout.size().rows() << "×" << pout.size().cols() << std::endl;
#endif
   // Transform encoder output posteriors to blockmodem priors (row-major order)
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

template <class dbl>
void map_reshape<dbl>::doinverse(const array2vd_t& pin,
      array2vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0, 0).size() == Base::M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   pout.init(this->input_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_reshape): Inverse ";
   libbase::trace << pin.size().rows() << "×" << pin.size().cols() << " to ";
   libbase::trace << pout.size().rows() << "×" << pout.size().cols() << std::endl;
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
std::string map_reshape<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Reshaping Mapper (Matrix) ";
   sout << " [" << this->input_block_size().rows() << "×"
         << this->input_block_size().cols() << " <-> "
         << this->output_block_size().rows() << "×"
         << this->output_block_size().cols() << "]";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_reshape<dbl>::serialize(
      std::ostream& sout) const
   {
   sout << size_out << std::endl;
   return sout;
   }

template <class dbl>
std::istream& map_reshape<dbl>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> size_out >> libbase::verify;
   return sin;
   }

} // end namespace

#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)

/* Serialization string: map_reshape<real>
 * where:
 *      real = float | double | logrealfast
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, x, type) \
      template class map_reshape<type>; \
      template <> \
      const serializer map_reshape<type>::shelper( \
            "mapper", \
            "map_reshape<" BOOST_PP_STRINGIZE(type) ">", \
            map_reshape<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
