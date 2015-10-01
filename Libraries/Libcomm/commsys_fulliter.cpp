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

#include "commsys_fulliter.h"
#include "vectorutils.h"
#include "vector_itfunc.h"
#include "modem/informed_modulator.h"
#include "codec/codec_softout.h"
#include "gf.h"

#include <sstream>
#include <typeinfo>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Log calls to receive_path and decode
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Communication System Interface

template <class S, template <class > class C>
void commsys_fulliter<S, C>::receive_path(const C<S>& received)
   {
#if DEBUG>=2
   libbase::trace << "DEBUG (fulliter): Starting receive path." << std::endl;
#endif
   // Store received vector
   last_received = received;
   // Reset modem
   ptable_ext_modem.init(0);
   ptable_ext_codec.init(0);
   cur_mdm_iter = 0;
   // Reset decoder
   cur_cdc_iter = 0;
   }

template <class S, template <class > class C>
void commsys_fulliter<S, C>::decode(C<int>& decoded)
   {
#if DEBUG>=2
   libbase::trace << "DEBUG (fulliter): Starting decode cycle " << cur_mdm_iter
   << "/" << cur_cdc_iter << "." << std::endl;
#endif
   // If this is the first decode cycle, we need to do the receive-path first
   if (cur_cdc_iter == 0)
      {
      // ** Inner code (modem class) **
      // Demodulate
      C<array1d_t> ptable_post_modem;
      informed_modulator<S>& m =
            dynamic_cast<informed_modulator<S>&> (*this->mdm);
      m.demodulate(*this->rxchan, last_received, ptable_ext_modem, ptable_post_modem);
      // Normalize posterior information
      libbase::normalize_results(ptable_post_modem, ptable_post_modem);
      // Inverse Map posterior information
      C<array1d_t> ptable_post_codec;
      this->map->inverse(ptable_post_modem, ptable_post_codec);
      // Compute extrinsic information from uncoded posteriors and priors
      // (codec alphabet)
      libbase::compute_extrinsic(ptable_ext_codec, ptable_post_codec,
            ptable_ext_codec);
      // Pass extrinsic information through mapper
      this->map->transform(ptable_ext_codec, ptable_ext_modem);
      // Mark mapper as clean (we will need to use again this cycle)
      this->map->mark_as_clean();

      // ** Outer code (codec class) **
      // Translate
      this->cdc->init_decoder(ptable_ext_codec);
      }
   // Perform soft-output decoding
   codec_softout<C>& c = dynamic_cast<codec_softout<C>&> (*this->cdc);
   C<array1d_t> ri_codec;
   C<array1d_t> ro_codec;
   c.softdecode(ri_codec, ro_codec);
   // Compute hard-decision for results gatherer
   hd_functor(ri_codec, decoded);
   // Compute feedback path if this is the last codec iteration
   if (++cur_cdc_iter == this->cdc->num_iter())
      {
      // Normalize posterior information
      libbase::normalize_results(ro_codec, ro_codec);
      // Pass posterior information through mapper
      C<array1d_t> ro_modem;
      this->map->transform(ro_codec, ro_modem);
      // Compute extrinsic information from encoded posteriors and priors
      // (modem alphabet)
      libbase::compute_extrinsic(ptable_ext_modem, ro_modem, ptable_ext_modem);
      // Inverse Map extrinsic information
      this->map->inverse(ptable_ext_modem, ptable_ext_codec);

      // Reset decoder iteration count
      cur_cdc_iter = 0;
      // Update modem iteration count
      cur_mdm_iter++;
      // If this was not the last iteration, mark components as clean
      if (cur_mdm_iter < iter)
         {
         this->mdm->mark_as_clean();
         this->map->mark_as_clean();
         }
      }
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_fulliter<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Full-System Iterative ";
   sout << Base::description() << ", ";
   sout << iter << " iterations";
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_fulliter<S, C>::serialize(std::ostream& sout) const
   {
   sout << "# Number of full-system iterations" << std::endl;
   sout << iter << std::endl;
   Base::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_fulliter<S, C>::serialize(std::istream& sin)
   {
   // read number of full-system iterations
   sin >> libbase::eatcomments >> iter >> libbase::verify;
   // next read underlying system
   Base::serialize(sin);
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::matrix;
using libbase::vector;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
   //(vector)(matrix)

/* Serialization string: commsys_fulliter<type,container>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class commsys_fulliter<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_fulliter<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "commsys", \
            "commsys_fulliter<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_fulliter<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
