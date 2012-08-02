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

#include "commsys_stream.h"

#include <sstream>

namespace libcomm {

// Communication System Interface

/*! \brief Determine SOF/EOF priors for next frame, given EOF post of this frame
 *
 * Determine start-of-frame and end-of-frame probabilities for the next frame,
 * given the (possibly empty) end-of-frame posterior probability for the
 * current frame.
 *
 * If the EOF post is supplied, the corresponding offset must also be supplied.
 * This is updated with the offset for the posterior probabilities.
 */
template <class S, template <class > class C>
void commsys_stream<S, C>::compute_priors(const C<double>& eof_post,
      C<double>& sof_prior, C<double>& eof_prior, libbase::size_type<C>& offset) const
   {
   // Shorthand for transmitted frame size
   const int tau = this->output_block_size();
   // Get access to the channel object in stream-oriented mode
   const channel_stream<S>& c = getrxchan_stream();

   if (eof_post.size() == 0) // this is the first frame
      {
      // Initialize as drift pdf after transmitting one frame
      c.get_drift_pdf(tau, eof_prior, offset);
      eof_prior /= eof_prior.max();
      // Initialize as zero-drift is assured
      sof_prior.init(eof_prior.size());
      sof_prior = 0;
      sof_prior(0 + offset) = 1;
      }
   else
      {
      // Use previous (centralized) end-of-frame posterior probability
      sof_prior = eof_post;
      // Initialize as drift pdf after transmitting one frame, given sof priors
      // (offset gets updated and sof_prior gets resized as needed)
      c.get_drift_pdf(tau, sof_prior, eof_prior, offset);
      eof_prior /= eof_prior.max();
      }
   }

template <class S, template <class > class C>
void commsys_stream<S, C>::receive_path(const C<S>& received,
      const C<double>& sof_prior, const C<double>& eof_prior,
      const libbase::size_type<C> offset)
   {
   // Get access to the commsys modem in stream-oriented mode
   stream_modulator<S, C>& m = getmodem_stream();
   // Demodulate
   C<array1d_t> ptable_mapped;
   m.reset_timers();
   m.demodulate(*this->rxchan, received, sof_prior, eof_prior, ptable_mapped,
         ptable_mapped, sof_post, eof_post, offset);
   this->add_timers(m);
   // After-demodulation receive path
   Base::softreceive_path(ptable_mapped);
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_stream<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Stream-oriented ";
   sout << Base::description();
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_stream<S, C>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_stream<S, C>::serialize(std::istream& sin)
   {
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

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
//(vector)(matrix)

/* Serialization string: commsys_stream<type>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 */
#define INSTANTIATE(r, args) \
      template class commsys_stream<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_stream<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "commsys", \
            "commsys_stream<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_stream<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ))

} // end namespace
