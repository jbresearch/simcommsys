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

#include "memoryless.h"
#include "fsm/cached_fsm.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class dbl>
void memoryless<dbl>::init()
   {
   assertalways(encoder);
   // Check that FSM is memoryless
   assertalways(encoder->mem_order() == 0);
   }

template <class dbl>
void memoryless<dbl>::free()
   {
   if (encoder != NULL)
      delete encoder;
   }

// constructor / destructor

template <class dbl>
memoryless<dbl>::memoryless(const fsm& encoder, const int tau) :
   tau(tau)
   {
   memoryless<dbl>::encoder = dynamic_cast<fsm*> (new cached_fsm(encoder));
   init();
   }

// internal codec operations

template <class dbl>
void memoryless<dbl>::resetpriors()
   {
   // Allocate space for prior input statistics
   libbase::allocate(rp, This::input_block_size(), This::num_inputs());
   // Initialize
   rp = 1.0;
   }

template <class dbl>
void memoryless<dbl>::setpriors(const array1vd_t& ptable)
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
void memoryless<dbl>::setreceiver(const array1vd_t& ptable)
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
void memoryless<dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Inherit sizes
   const int k = enc_inputs();
   const int n = enc_outputs();
   // Reset the encoder to zero state
   encoder->reset();
   // Initialise result vector
   encoded.init(This::output_block_size());
   // Encode source stream
   for (int t = 0; t < tau; t++)
      {
      // have to make a copy because of step()
      array1i_t ip = source.extract(t * k, k);
      encoded.segment(t * n, n) = encoder->step(ip);
      }
   }

template <class dbl>
void memoryless<dbl>::softdecode(array1vd_t& ri)
   {
   // Inherit sizes
   const int k = enc_inputs();
   const int n = enc_outputs();
   const int S = This::num_inputs();
   // Initialize results to prior statistics
   ri = rp;
   // Consider each time-step
   for (int t = 0; t < tau; t++)
      {
      array1i_t ip(k), op(n);
      // Go through each possible input set
      ip = 0;
      for (int i = 0; i < k; i++)
         for (int x = 0; x < S; x++)
            {
            // Update input set
            ip(i) = x;
            // Determine corresponding output set
            op = encoder->step(ip);
            // Update probabilities at input according to those at output
            for (int ii = 0; ii < k; ii++)
               for (int oo = 0; oo < n; oo++)
                  ri(t * k + ii)(ip(ii)) *= R(t * n + oo)(op(oo));
            }
      }
   }

template <class dbl>
void memoryless<dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   softdecode(ri);
   // Inherit sizes
   const int k = enc_inputs();
   const int n = enc_outputs();
   const int S = This::num_inputs();
   // Allocate space for output results
   libbase::allocate(ro, This::output_block_size(), This::num_outputs());
   // Initialize
   ro = 1.0;
   // Consider each time-step
   for (int t = 0; t < tau; t++)
      {
      array1i_t ip(k), op(n);
      // Go through each possible input set
      ip = 0;
      for (int i = 0; i < k; i++)
         for (int x = 0; x < S; x++)
            {
            // Update input set
            ip(i) = x;
            // Determine corresponding output set
            op = encoder->step(ip);
            // Update probabilities at output according to those at input
            for (int ii = 0; ii < k; ii++)
               for (int oo = 0; oo < n; oo++)
                  ro(t * n + oo)(op(oo)) *= ri(t * k + ii)(ip(ii));
            }
      }
   }

// description output

template <class dbl>
std::string memoryless<dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Memoryless Code (" << This::output_bits() << ","
         << This::input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class dbl>
std::ostream& memoryless<dbl>::serialize(std::ostream& sout) const
   {
   sout << "# Encoder" << std::endl;
   sout << encoder;
   sout << "# Block length" << std::endl;
   sout << tau << std::endl;
   return sout;
   }

// object serialization - loading

template <class dbl>
std::istream& memoryless<dbl>::serialize(std::istream& sin)
   {
   free();
   // load the encoder into a temporary space
   fsm *encoder_original;
   sin >> libbase::eatcomments >> encoder_original >> libbase::verify;
   // see if this is already a cached fsm;
   cached_fsm *encoder_cached = dynamic_cast<cached_fsm*> (encoder_original);
   if (encoder_cached)
      // if it is, just make a copy
      memoryless<dbl>::encoder = dynamic_cast<fsm*> (encoder_original->clone());
   else
      // otherwise, create a cached fsm copy on the fly
      memoryless<dbl>::encoder = dynamic_cast<fsm*> (new cached_fsm(
            *encoder_original));
   // clean up
   delete encoder_original;
   // read the block length
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   init();
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

/* Serialization string: memoryless<real>
 * where:
 *      real = float | double | mpreal | mpgnu | logreal | logrealfast
 */
#define INSTANTIATE(r, x, type) \
      template class memoryless<type>; \
      template <> \
      const serializer memoryless<type>::shelper( \
            "codec", \
            "memoryless<" BOOST_PP_STRINGIZE(type) ">", \
            memoryless<type>::create); \

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
