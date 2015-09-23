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

#include "commsys_stream.h"

#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of encoded/decoded frame sizes and drift
// 3 - Observe prior and posterior PDF's
// 4 - For fidelity collector, observe actual/estimated boundary drifts
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Communication System Interface

/*! \brief Advance the stream, skipping over last decoded frame
 * \param[in] recevied      The received stream (to be updated)
 * \param[in] oldoffset     The offset value that applied for the last decoding
 * \param[in] drift         The estimated drift at the end of decoded frame
 * \param[in] newoffset     The offset value that applies for the next decoding
 *
 * This method initializes the received stream, if empty, by the new offset.
 * Otherwise, it will skip over all material from the last decoding that is
 * no longer needed (keeping the last segment as required by the new offset).
 */
template <class S, template <class > class C, class real>
void commsys_stream<S, C, real>::stream_advance(C<S>& received,
      const libbase::size_type<C>& oldoffset,
      const libbase::size_type<C>& drift,
      const libbase::size_type<C>& newoffset)
   {
   C<S> received_prev;
   if (received.size() == 0)
      {
      received_prev.init(newoffset);
      received_prev = 0; // value is irrelevant as this is not used
      }
   else
      {
      const int tau = this->output_block_size();
      const int start = tau + drift + oldoffset - newoffset;
      const int length = received.size() - start;
      received_prev = received.extract(start, length);
      }
   received = received_prev;
   // Tell user what we're doing
#if DEBUG>=2
   std::cerr << "DEBUG (commsys_stream): old offset = " << oldoffset << std::endl;
   std::cerr << "DEBUG (commsys_stream): frame drift = " << drift << std::endl;
   std::cerr << "DEBUG (commsys_stream): new offset = " << newoffset << std::endl;
   std::cerr << "DEBUG (commsys_stream): existing segment = " << received.size() << std::endl;
#endif
   }

/*! \brief Determine SOF/EOF priors for next frame, given EOF post of this frame
 *         and the look-ahead quantity required
 *
 * Determine start-of-frame and end-of-frame probabilities for the next frame
 * (plus the specified look-ahead quantity), given the (possibly empty)
 * end-of-frame posterior probability for the current frame.
 *
 * If the EOF post is supplied, the corresponding offset must also be supplied.
 * This is updated with the offset for the posterior probabilities.
 */
template <class S, template <class > class C, class real>
void commsys_stream<S, C, real>::compute_priors(const C<double>& eof_post,
      const libbase::size_type<C> lookahead, C<double>& sof_prior,
      C<double>& eof_prior, libbase::size_type<C>& offset) const
   {
   // Shorthand for transmitted frame size + required look-ahead
   const int tau = this->output_block_size() + lookahead;
   // Shorthand for probability of channel event outside chosen limits
   const double Pr = getmodem_stream().get_suggested_exclusion();
   // Get access to the channel object in stream-oriented mode
   const channel_stream<S, real>& rxchan = getrxchan_stream();

   if (eof_post.size() == 0) // this is the first frame
      {
      // Initialize as drift pdf after transmitting one frame
      rxchan.get_drift_pdf(tau, Pr, eof_prior, offset);
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
      rxchan.get_drift_pdf(tau, Pr, sof_prior, eof_prior, offset);
      eof_prior /= eof_prior.max();
      }
   }

template <class S, template <class > class C, class real>
void commsys_stream<S, C, real>::receive_path(const C<S>& received,
      const libbase::size_type<C> lookahead, const C<double>& sof_prior,
      const C<double>& eof_prior, const libbase::size_type<C> offset)
   {
   // Get access to the commsys modem in stream-oriented mode
   stream_modulator<S, C>& mdm = getmodem_stream();
   // Demodulate
   C<array1d_t> ptable_mapped;
   mdm.reset_timers();
   mdm.demodulate(*this->rxchan, received, lookahead, sof_prior, eof_prior,
         ptable_mapped, ptable_mapped, sof_post, eof_post, offset);
   this->add_timers(mdm);
   // After-demodulation receive path
   Base::softreceive_path(ptable_mapped);
   }

// Description & Serialization

template <class S, template <class > class C, class real>
std::string commsys_stream<S, C, real>::description() const
   {
   std::ostringstream sout;
   sout << "Stream-oriented ";
   sout << Base::description();
   if (iter > 1)
      sout << ", " << iter << " full-system iterations";
   return sout.str();
   }

// object serialization - saving

template <class S, template <class > class C, class real>
std::ostream& commsys_stream<S, C, real>::serialize(std::ostream& sout) const
   {
   // first write underlying system
   Base::serialize(sout);
   // format version
   sout << "# Version (stream extensions)" << std::endl;
   sout << 1 << std::endl;
   sout << "# Number of full-system iterations" << std::endl;
   sout << iter << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added number of full-system iterations
 */

template <class S, template <class > class C, class real>
std::istream& commsys_stream<S, C, real>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   // first read underlying system
   Base::serialize(sin);
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files
   if (sin.fail())
      {
      version = 0;
      sin.clear();
      }
   // reset (valid for version 0)
   iter = 1;
   // read number of full-system iterations
   if (version >= 1)
      sin >> libbase::eatcomments >> iter >> libbase::verify;
   // check that components are stream-oriented
   getmodem_stream();
   getrxchan_stream();
   gettxchan_stream();
   getcodec_softout();
   // we're done
   assertalways(sin.good());
   return sin;
   }

} // end namespace

#include "gf.h"
#include "erasable.h"
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::erasable;
using libbase::matrix;
using libbase::vector;
using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define FINITE_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ

#define ADD_ERASABLE(r, x, type) \
   (type)(erasable<type>)

#define ALL_FINITE_TYPE_SEQ \
   BOOST_PP_SEQ_FOR_EACH(ADD_ERASABLE, x, FINITE_TYPE_SEQ)

// *** Stream Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace) \
   ALL_FINITE_TYPE_SEQ
#define CONTAINER_TYPE_SEQ \
   (vector)
//(vector)(matrix)
#ifdef USE_CUDA
#define REAL_TYPE_SEQ \
   (float)(double)
#else
#define REAL_TYPE_SEQ \
   (float)(double)(mpgnu)(logrealfast)
#endif

/* Serialization string: commsys_stream<type,container,real>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 *      real = float | double | [mpgnu | logrealfast (CPU only)]
 */
#define INSTANTIATE(r, args) \
      template class commsys_stream<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_stream<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "commsys", \
            "commsys_stream<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            commsys_stream<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
