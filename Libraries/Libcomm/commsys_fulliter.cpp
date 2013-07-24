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

/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
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
// 3 - Show part of soft information being passed around
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
   ptable_mapped.init(0);
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
      // Demodulate
      C<array1d_t> ptable_full;
      informed_modulator<S>& m =
            dynamic_cast<informed_modulator<S>&> (*this->mdm);
      m.demodulate(*this->rxchan, last_received, ptable_mapped, ptable_full);
#if DEBUG>=3
      libbase::trace << "DEBUG (fulliter): modem soft-output = " << std::endl;
      libbase::trace << ptable_mapped.extract(0,5);
#endif
      // Compute extrinsic information for passing to codec
      libbase::compute_extrinsic(ptable_mapped, ptable_full, ptable_mapped);
      // After-demodulation receive path
      this->softreceive_path(ptable_mapped);
      }
   // Just do a plain decoder iteration if this is not the last one in the cycle
   if (++cur_cdc_iter < this->cdc->num_iter())
      this->cdc->decode(decoded);
   // Otherwise, do a soft-output iteration
   else
      {
      // Perform soft-output decoding
      codec_softout<C>& c = dynamic_cast<codec_softout<C>&> (*this->cdc);
      C<array1d_t> ri;
      C<array1d_t> ro;
      c.softdecode(ri, ro);
      // Compute hard-decision for results gatherer
      hd_functor(ri, decoded);
      // Pass posterior information through mapper
      C<array1d_t> ro_mapped;
      this->map->transform(ro, ro_mapped);
      // Compute extrinsic information for next demodulation cycle
      libbase::compute_extrinsic(ptable_mapped, ro_mapped, ptable_mapped);
#if DEBUG>=3
      libbase::trace << "DEBUG (fulliter): codec soft-output = " << std::endl;
      libbase::trace << ptable_mapped.extract(0,5);
#endif
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
   sout << iter;
   Base::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_fulliter<S, C>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter >> libbase::verify;
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
