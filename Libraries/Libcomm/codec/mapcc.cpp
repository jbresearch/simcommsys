/*!
 * \file
 * $Id$
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

#include "mapcc.h"
#include "mapper/map_straight.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// initialization / de-allocation

template <class real, class dbl>
void mapcc<real, dbl>::init()
   {
   assertalways(encoder);
   BCJR::init(*encoder, tau);
   assertalways(!circular || !endatzero);
   }

template <class real, class dbl>
void mapcc<real, dbl>::free()
   {
   if (encoder != NULL)
      delete encoder;
   }

template <class real, class dbl>
void mapcc<real, dbl>::reset()
   {
   if (circular)
      {
      BCJR::setstart();
      BCJR::setend();
      }
   else if (endatzero)
      {
      BCJR::setstart(0);
      BCJR::setend(0);
      }
   else
      {
      BCJR::setstart(0);
      BCJR::setend();
      }
   }

// constructor / destructor

template <class real, class dbl>
mapcc<real, dbl>::mapcc() :
   encoder(NULL)
   {
   }

template <class real, class dbl>
mapcc<real, dbl>::mapcc(const fsm& encoder, const int tau,
      const bool endatzero, const bool circular) :
   tau(tau), endatzero(endatzero), circular(circular)
   {
   This::encoder = dynamic_cast<fsm*> (encoder.clone());
   init();
   }

// internal codec functions

template <class real, class dbl>
void mapcc<real, dbl>::resetpriors()
   {
   // Initialize input probability vector
   app.init(BCJR::block_size(), BCJR::num_input_symbols());
   app = 1.0;
   }

template <class real, class dbl>
void mapcc<real, dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Convert the input statistics for the BCJR Algorithm

   // Set up mapper
   map_straight<libbase::vector, dbl> map;
   const int N = BCJR::num_input_symbols(); // codec output alphabet
   const int M = This::num_inputs(); // blockmodem input alphabet
   map.set_parameters(N, M);
   map.set_blocksize(BCJR::block_size());
   // Convert to a temporary space
   array1vd_t ptable_bcjr;
   map.inverse(ptable, ptable_bcjr);
   // Initialize input probability vector
   app.init(BCJR::block_size(), BCJR::num_input_symbols());
   // Copy the input statistics for the BCJR Algorithm
   for (int t = 0; t < app.size().rows(); t++)
      for (int i = 0; i < app.size().cols(); i++)
         app(t, i) = ptable_bcjr(t)(i);
   }

template <class real, class dbl>
void mapcc<real, dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Confirm input alphabet size same as encoded alphabet
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());
   // Inherit sizes
   const int S = encoder->num_symbols();
   const int N = encoder->num_output_combinations();
   const int n = encoder->num_outputs();
   // Initialize receiver probability vector
   R.init(tau, N);
   // Copy the input statistics for the BCJR Algorithm
   R = 1.0;
   for (int t = 0; t < tau; t++)
      for (int x = 0; x < N; x++)
         for (int i = 0, thisx = x; i < n; i++, thisx /= S)
            R(t, x) *= dbl(ptable(t * n + i)(thisx % S));
   // Reset start- and end-state probabilities
   reset();
   }

// encoding and decoding functions

template <class real, class dbl>
void mapcc<real, dbl>::do_encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Inherit sizes
   const int k = encoder->num_inputs();
   const int n = encoder->num_outputs();
   // Reform source into a matrix, with one row per timestep
   // and adding any necessary tail
   array2i_t source1(tau, k);
   source1 = fsm::tail;
   source1.copyfrom(source);
   // Reset the encoder to zero state
   encoder->reset();
   // When dealing with a circular system, perform first pass to determine end
   // state, then reset to the corresponding circular state.
   if (circular)
      {
      for (int t = 0; t < tau; t++)
         {
         array1i_t ip = source1.extractrow(t);
         encoder->advance(ip);
         }
      encoder->resetcircular();
      }
   // Initialise result vector
   array2i_t encoded1(tau, n);
   // Encode source stream
   for (int t = 0; t < tau; t++)
      {
      array1i_t ip = source1.extractrow(t);
      encoded1.insertrow(encoder->step(ip), t);
      source1.insertrow(ip, t);
      }
   // Reform results as a vector
   encoded = encoded1.rowmajor();
   }

template <class real, class dbl>
void mapcc<real, dbl>::softdecode(array1vd_t& ri)
   {
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif;
   // perform decoding
   BCJR::fdecode(R, app, rif);
   // initialise results set
   libbase::allocate(ri, This::input_block_size(), This::num_inputs());
   // remove any tail bits from input set
   for (int i = 0; i < This::input_block_size(); i++)
      for (int j = 0; j < This::num_inputs(); j++)
         ri(i)(j) = rif(i, j);
   }

template <class real, class dbl>
void mapcc<real, dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   // temporary space to hold complete results (ie. with tail)
   array2d_t rif, rof;
   // perform decoding
   BCJR::decode(R, rif, rof);
   // remove any tail bits from input set
   libbase::allocate(ri, This::input_block_size(), This::num_inputs());
   for (int i = 0; i < This::input_block_size(); i++)
      for (int j = 0; j < This::num_inputs(); j++)
         ri(i)(j) = rif(i, j);
   // copy output set
   libbase::allocate(ro, This::output_block_size(), This::num_outputs());
   for (int i = 0; i < This::output_block_size(); i++)
      for (int j = 0; j < This::num_outputs(); j++)
         ro(i)(j) = rof(i, j);
   }

// description output

template <class real, class dbl>
std::string mapcc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << (endatzero ? "Terminated, " : "Unterminated, ");
   sout << (circular ? "Circular, " : "Non-circular, ");
   sout << "MAP-decoded Convolutional Code (" << This::output_bits() << ","
         << This::input_bits() << ") - ";
   sout << encoder->description();
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& mapcc<real, dbl>::serialize(std::ostream& sout) const
   {
   sout << encoder;
   sout << tau << std::endl;
   sout << int(endatzero) << std::endl;
   sout << int(circular) << std::endl;
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& mapcc<real, dbl>::serialize(std::istream& sin)
   {
   free();
   sin >> libbase::eatcomments >> encoder >> libbase::verify;
   sin >> libbase::eatcomments >> tau >> libbase::verify;
   sin >> libbase::eatcomments >> endatzero >> libbase::verify;
   sin >> libbase::eatcomments >> circular >> libbase::verify;
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
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL1_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double) \
   (logrealfast)

/* Serialization string: mapcc<real1,real2>
 * where:
 *      real1 = float | double | mpreal | mpgnu | logreal | logrealfast
 *              [real1 is the internal arithmetic type]
 *      real2 = float | double | logrealfast
 *              [real2 is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class mapcc<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer mapcc<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "mapcc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            mapcc<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL1_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
