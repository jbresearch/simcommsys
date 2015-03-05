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

#include "uncoded.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// encoding and decoding functions

template <class dbl>
void uncoded<dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
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
void uncoded<dbl>::do_init_decoder(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Copy the received (output-referred) statistics
   R = ptable;
   }

template <class dbl>
void uncoded<dbl>::do_init_decoder(const array1vd_t& ptable, const array1vd_t& app)
   {
   // Initialize results to received statistics
   do_init_decoder(ptable);
   // Multiply with prior statistics
   R *= app;
   }

template <class dbl>
void uncoded<dbl>::softdecode(array1vd_t& ri)
   {
   // Set input-referred statistics to stored values
   ri = R;
   }

template <class dbl>
void uncoded<dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   // Set input-referred statistics to stored values
   ri = R;
   // Set output-referred statistics to stored values
   ro = R;
   }

// description output

template <class dbl>
std::string uncoded<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Uncoded Representation (" << N << "×" << q << ")";
   return sout.str();
   }

// object serialization - saving

template <class dbl>
std::ostream& uncoded<dbl>::serialize(std::ostream& sout) const
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
std::istream& uncoded<dbl>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
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
      template class uncoded<type>; \
      template <> \
      const serializer uncoded<type>::shelper( \
            "codec", \
            "uncoded<" BOOST_PP_STRINGIZE(type) ">", \
            uncoded<type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
