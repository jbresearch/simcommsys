/*!
 * \file
 * $Id: uncoded.cpp 9909 2013-09-23 08:43:23Z jabriffa $
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

#include "conv.h"
//#include "uncoded.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// internal codec operations

template <class dbl>
void conv<dbl>::resetpriors()
   {
   // Allocate space for prior input statistics
   libbase::allocate(rp, This::input_block_size(), This::num_inputs());
   // Initialize
   rp = 1.0;
   }

template <class dbl>
void conv<dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Copy the input statistics
   rp = ptable;
   }

template <class dbl>
void conv<dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Copy the output statistics
   R = ptable;
   }

// encoding and decoding functions

template <class dbl>
void conv<dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
#ifndef NDEBUG
   // Validate input
   assert(source.size() == N);
   for (int i = 0; i < N; i++)
      assert(source(i) >= 0 && source(i) < q);
#endif
   // Copy input to output
   encoded = source;
   }

template <class dbl>
void conv<dbl>::softdecode(array1vd_t& ri)
   {
   // Initialize results to prior statistics
   ri = rp;
   // Multiply with received statistics
   ri *= R;
   }

template <class dbl>
void conv<dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   // Determine input-referred statistics
   softdecode(ri);
   // Copy output-referred statistics from input-referred ones
   ro = ri;
   }

// description output

template <class dbl>
std::string conv<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded Representation (" << N << "x" << q << ")";
   return sout.str();
   }

// object serialization - saving

template <class dbl>
std::ostream& conv<dbl>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Alphabet size" << std::endl;
   sout << q << std::endl;
   sout << "# Block length" << std::endl;
   sout << N << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */
template <class dbl>
std::istream& conv<dbl>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // read the alphabet size and block length
   sin >> libbase::eatcomments >> q >> libbase::verify;
   sin >> libbase::eatcomments >> N >> libbase::verify;
   return sin;
   }

} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)

/* Serialization string: uncoded<real>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE(r, x, type) \
      template class conv<type>; \
      template <> \
      const serializer conv<type>::shelper( \
            "codec", \
            "conv<" BOOST_PP_STRINGIZE(type) ">", \
            conv<type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
