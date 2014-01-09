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

#include "codec_concatenated.h"
#include "vector_itfunc.h"
#include <sstream>

namespace libcomm {

// internal operations

template <template <class > class C, class dbl>
void codec_concatenated<C, dbl>::init()
   {
   test_invariant();
   // iterators for codec and mapper to use
   typename codec_list_t::iterator codec_it = codec_list.begin();
   typename mapper_list_t::iterator mapper_it = mapper_list.begin();
   // repeat for all mappers
   for(; mapper_it != mapper_list.end(); mapper_it++)
      {
      // determine mapper input parameters (from prior codec)
      const libbase::size_type<C> N = (*codec_it)->output_block_size();
      const int q = (*codec_it)->num_outputs();
      // determine mapper output parameters (from posterior codec)
      codec_it++;
      const int M = (*codec_it)->num_inputs();
      // set up mapper with required parameters
      (*mapper_it)->set_parameters(q, M);
      (*mapper_it)->set_blocksize(N);
      }
   test_invariant();
   }

// encode / decode methods

template <template <class > class C, class dbl>
void codec_concatenated<C, dbl>::do_encode(const C<int>& source, C<int>& encoded)
   {
   test_invariant();
   // iterators for codec and mapper to use
   typename codec_list_t::iterator codec_it = codec_list.begin();
   typename mapper_list_t::iterator mapper_it = mapper_list.begin();
   // placeholders for intermediate results
   C<int> a = source;
   C<int> b;
   // pass through all codec+mapper combinations (everything before last codec)
   for(; mapper_it != mapper_list.end(); mapper_it++, codec_it++)
      {
      (*codec_it)->encode(a, b);
      (*mapper_it)->transform(b, a);
      }
   // pass through last codec
   (*codec_it)->encode(a, b);
   // copy result
   encoded = b;
   test_invariant();
   }

template <template <class > class C, class dbl>
void codec_concatenated<C, dbl>::softdecode(C<array1d_t>& ri)
   {
   test_invariant();
   // reverse iterators for codec and mapper to use
   typename codec_list_t::reverse_iterator codec_it = codec_list.rbegin();
   typename mapper_list_t::reverse_iterator mapper_it = mapper_list.rbegin();
   // placeholders for intermediate results
   C<array1d_t> ri_codec, ri_mapper;
   // pass through first codec
   for (int i = 0; i < (*codec_it)->num_iter(); i++)
      (*codec_it)->softdecode(ri_codec);
   // pass through all mapper+codec combinations (everything after first codec)
   for(codec_it++; mapper_it != mapper_list.rend(); mapper_it++, codec_it++)
      {
      // Normalize posterior information
      libbase::normalize_results(ri_codec, ri_codec);
      // Pass posterior information through mapper
      (*mapper_it)->inverse(ri_codec, ri_mapper);
      // Initialize decoder
      (*codec_it)->init_decoder(ri_mapper);
      // Perform soft-output decoding
      for (int i = 0; i < (*codec_it)->num_iter(); i++)
         (*codec_it)->softdecode(ri_codec);
      }
   // copy result
   ri = ri_codec;
   test_invariant();
   }

template <template <class > class C, class dbl>
void codec_concatenated<C, dbl>::softdecode(C<array1d_t>& ri, C<array1d_t>& ro)
   {
   test_invariant();
   // reverse iterators for codec and mapper to use
   typename codec_list_t::reverse_iterator codec_it = codec_list.rbegin();
   typename mapper_list_t::reverse_iterator mapper_it = mapper_list.rbegin();
   // placeholders for intermediate results
   C<array1d_t> ri_codec, ro_codec, ri_mapper;
   // pass through first codec
   for (int i = 0; i < (*codec_it)->num_iter(); i++)
      (*codec_it)->softdecode(ri_codec);
   // pass through all mapper+codec combinations (everything after first codec)
   for(codec_it++; mapper_it != mapper_list.rend(); mapper_it++, codec_it++)
      {
      // Normalize posterior information
      libbase::normalize_results(ri_codec, ri_codec);
      // Pass posterior information through mapper
      (*mapper_it)->inverse(ri_codec, ri_mapper);
      // Initialize decoder
      (*codec_it)->init_decoder(ri_mapper);
      // Perform soft-output decoding
      for (int i = 0; i < (*codec_it)->num_iter(); i++)
         (*codec_it)->softdecode(ri_codec, ro_codec);
      }
   // copy result
   ri = ri_codec;
   ro = ro_codec;
   test_invariant();
   }

// description output

template <template <class > class C, class dbl>
std::string codec_concatenated<C, dbl>::description() const
   {
   test_invariant();
   std::ostringstream sout;
   sout << "Concatenated codec (" << codec_list.size() << " codecs, "
         << mapper_list.size() << " mappers) [";
   size_t i;
   // codecs description
   i = 0;
   for(typename codec_list_t::const_iterator it = codec_list.begin(); it != codec_list.end(); it++)
      {
      sout << "C" << ++i << ": " << (*it)->description();
      if (i < codec_list.size())
         sout << ", ";
      }
   // mappers description
   if (mapper_list.size() > 0)
      sout << ", ";
   i = 0;
   for(typename mapper_list_t::const_iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
      {
      sout << "M" << ++i << ": " << (*it)->description();
      if (i < mapper_list.size())
         sout << ", ";
      }
   sout << "]";
   return sout.str();
   }

// object serialization - saving

template <template <class > class C, class dbl>
std::ostream& codec_concatenated<C, dbl>::serialize(std::ostream& sout) const
   {
   test_invariant();
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Number of concatenated codecs" << std::endl;
   sout << codec_list.size() << std::endl;
   size_t i;
   // serialize codecs
   i = 0;
   for(typename codec_list_t::const_iterator it = codec_list.begin(); it != codec_list.end(); it++)
      {
      sout << "# Codec " << ++i << std::endl;
      sout << *it;
      }
   // serialize mappers
   i = 0;
   for(typename mapper_list_t::const_iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
      {
      sout << "# Mapper " << ++i << std::endl;
      sout << *it;
      }
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */
template <template <class > class C, class dbl>
std::istream& codec_concatenated<C, dbl>::serialize(std::istream& sin)
   {
   free();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // get number of codecs
   int N;
   sin >> libbase::eatcomments >> N >> libbase::verify;
   assertalways(N >= 1);
   // serialize codecs
   for(int i = 0; i < N; i++)
      {
      codec<C, dbl> *this_codec;
      sin >> libbase::eatcomments >> this_codec >> libbase::verify;
      // get access to soft-out object (and confirm this is valid)
      codec_softout<C, dbl> *this_codec_softout = dynamic_cast<codec_softout<C,
            dbl> *>(this_codec);
      assertalways(this_codec_softout);
      codec_list.push_back(this_codec_softout);
      }
   // serialize mappers
   for(int i = 0; i < N-1; i++)
      {
      mapper<C, dbl> *this_mapper;
      sin >> libbase::eatcomments >> this_mapper >> libbase::verify;
      mapper_list.push_back(this_mapper);
      }
   // initialize
   init();
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

/* Serialization string: codec_concatenated<real>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE(r, x, type) \
      template class codec_concatenated<libbase::vector, type>; \
      template <> \
      const serializer codec_concatenated<libbase::vector, type>::shelper( \
            "codec", \
            "codec_concatenated<" BOOST_PP_STRINGIZE(type) ">", \
            codec_concatenated<libbase::vector, type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, (double))
//BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
