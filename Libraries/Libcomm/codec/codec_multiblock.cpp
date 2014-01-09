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

#include "codec_multiblock.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// encode / decode methods

template <template <class > class C, class dbl>
void codec_multiblock<C, dbl>::do_encode(const C<int>& source, C<int>& encoded)
   {
   test_invariant();
   // allocate output vector
   encoded.init(this->output_block_size());
   // iterate over all blocks in sequence
   for (int i = 0; i < N; i++)
      {
      libbase::indirect_vector<int> src = source.extract(
            cdc->input_block_size() * i, cdc->input_block_size());
      libbase::indirect_vector<int> enc = encoded.segment(
            cdc->output_block_size() * i, cdc->output_block_size());
      cdc_enc->encode(src, enc);
      }
   test_invariant();
   }

template <template <class > class C, class dbl>
void codec_multiblock<C, dbl>::softdecode(C<array1d_t>& ri)
   {
   test_invariant();
   // allocate output vector
   libbase::allocate(ri, this->input_block_size(), this->num_inputs());
   // iterate over all blocks in sequence
   for (int i = 0; i < N; i++)
      {
      // Initialize the codec
      libbase::indirect_vector<array1d_t> ptable_segment = ptable.extract(
            cdc->output_block_size() * i, cdc->output_block_size());
      if (app.size() > 0)
         {
         libbase::indirect_vector<array1d_t> app_segment = app.extract(
               cdc->input_block_size() * i, cdc->input_block_size());
         cdc->init_decoder(ptable_segment, app_segment);
         }
      else
         cdc->init_decoder(ptable_segment);
      // Perform soft-output decoding for as many iterations as needed
      for (int j = 0; j < cdc->num_iter(); j++)
         {
         libbase::indirect_vector<array1d_t> ri_segment = ri.segment(
               cdc->input_block_size() * i, cdc->input_block_size());
         cdc->softdecode(ri_segment);
         }
      }
   test_invariant();
   }

template <template <class > class C, class dbl>
void codec_multiblock<C, dbl>::softdecode(C<array1d_t>& ri, C<array1d_t>& ro)
   {
   test_invariant();
   // allocate output vectors
   libbase::allocate(ri, this->input_block_size(), this->num_inputs());
   libbase::allocate(ro, this->output_block_size(), this->num_outputs());
   // iterate over all blocks in sequence
   for (int i = 0; i < N; i++)
      {
      // Initialize the codec
      libbase::indirect_vector<array1d_t> ptable_segment = ptable.extract(
            cdc->output_block_size() * i, cdc->output_block_size());
      if (app.size() > 0)
         {
         libbase::indirect_vector<array1d_t> app_segment = app.extract(
               cdc->input_block_size() * i, cdc->input_block_size());
         cdc->init_decoder(ptable_segment, app_segment);
         }
      else
         cdc->init_decoder(ptable_segment);
      // Perform soft-output decoding for as many iterations as needed
      for (int j = 0; j < cdc->num_iter(); j++)
         {
         libbase::indirect_vector<array1d_t> ri_segment = ri.segment(
               cdc->input_block_size() * i, cdc->input_block_size());
         libbase::indirect_vector<array1d_t> ro_segment = ro.segment(
               cdc->output_block_size() * i, cdc->output_block_size());
         cdc->softdecode(ri_segment, ro_segment);
         }
      }
   test_invariant();
   }

// description output

template <template <class > class C, class dbl>
std::string codec_multiblock<C, dbl>::description() const
   {
   test_invariant();
   std::ostringstream sout;
   sout << "Multi-block codec (" << N << " blocks) [";
   sout << cdc->description();
   sout << "]";
   return sout.str();
   }

// object serialization - saving

template <template <class > class C, class dbl>
std::ostream& codec_multiblock<C, dbl>::serialize(std::ostream& sout) const
   {
   test_invariant();
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Underlying codec" << std::endl;
   sout << cdc;
   sout << "# Number of blocks to aggregate" << std::endl;
   sout << N << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */
template <template <class > class C, class dbl>
std::istream& codec_multiblock<C, dbl>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // serialize codec
   codec<C, dbl> *this_codec;
   sin >> libbase::eatcomments >> this_codec >> libbase::verify;
   // get access to soft-out object (and confirm this is valid)
   cdc.reset(dynamic_cast<codec_softout<C, dbl> *>(this_codec));
   assertalways(cdc);
   // get number of blocks to aggregate
   sin >> libbase::eatcomments >> N >> libbase::verify;
   test_invariant();
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

/* Serialization string: codec_multiblock<real>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE(r, x, type) \
      template class codec_multiblock<libbase::vector, type>; \
      template <> \
      const serializer codec_multiblock<libbase::vector, type>::shelper( \
            "codec", \
            "codec_multiblock<" BOOST_PP_STRINGIZE(type) ">", \
            codec_multiblock<libbase::vector, type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, (double))
//BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
