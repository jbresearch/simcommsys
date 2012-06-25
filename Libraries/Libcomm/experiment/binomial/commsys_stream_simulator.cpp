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

#include "commsys_stream_simulator.h"

#include "commsys_stream.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of encoded/decoded frame sizes and drift
// 3 - Observe prior and posterior PDF's
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 3
#endif

// Experiment handling

/*!
 * \brief Perform a complete encode->transmit->receive cycle
 * \param[out] result   Vector containing the set of results to be updated
 *
 * Results are organized according to the collector used, as a function of
 * the iteration count.
 *
 * \note The results collector assumes that the result vector is an accumulator,
 * so that every call adds to the existing result. This explains the need to
 * initialize the result vector to zero.
 */
template <class S, class R>
void commsys_stream_simulator<S, R>::sample(libbase::vector<double>& result)
   {
   assert(sys_enc);

   // reset if we have reached the user-set limit for stream length
   if (N > 0 && frames_decoded >= N)
      reset();

   // Shorthand for transmitted frame size
   const int tau = this->sys->output_block_size();
   // Get access to the commsys channel object in stream-oriented mode
   const channel_stream<S>& c =
         dynamic_cast<const channel_stream<S>&> (*this->sys->getchan());

   // Keep a copy of the last frame's offset (in case x_max changes)
   const libbase::size_type<libbase::vector> oldoffset = offset;

   // Determine start-of-frame and end-of-frame probabilities
   libbase::vector<double> sof_prior;
   libbase::vector<double> eof_prior;
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

   // Extract required segment of existing stream
   libbase::vector<S> received_prev;
   if (received.size() == 0)
      {
      received_prev.init(offset);
      received_prev = 0; // value is irrelevant as this is not used
      }
   else
      {
      const int start = tau + estimated_drift + oldoffset - offset;
      const int length = received.size() - start;
      received_prev = received.extract(start, length);
      }
   received = received_prev;
   // Tell user what we're doing
#if DEBUG>=2
   std::cerr << "DEBUG (commsys_stream_simulator): est drift = "
         << estimated_drift << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): old offset = " << oldoffset
         << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): new offset = " << offset
         << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): existing segment = "
         << received.size() << std::endl;
#endif
   // Determine required segment size
   const int length = tau + eof_prior.size() - 1;
#if DEBUG>=2
   std::cerr << "DEBUG (commsys_stream_simulator): final segment size = "
         << length << std::endl;
#endif
   // Append required segment by simulating frame transmission
   // Also make sure that we have transmitted the frame corresponding to the
   // received one.
   for (int left = length - received.size(); left > 0 || source.empty();)
      {
      // Create next source frame
      const libbase::vector<int> source_next = Base::createsource();
      // Encode -> Map -> Modulate next frame
      const libbase::vector<S> transmitted = sys_enc->encode_path(source_next);
      // Transmit next frame
      const libbase::vector<S> received_next = this->sys->transmit(transmitted);
      // shorthand for received frame size
      const int rho = received_next.size();
      // store what we need to keep
      source.push_back(source_next);
      received = concatenate(received, received_next);
      actual_drift.push_back(rho - tau);
      // update counters
      frames_encoded++;
      left -= rho;
      // Tell user what we're doing
#if DEBUG>=2
      std::cerr << "DEBUG (commsys_stream_simulator): Actual received frame = "
            << rho << std::endl;
      std::cerr << "DEBUG (commsys_stream_simulator): Remaining length = "
            << left << std::endl;
      std::cerr << "DEBUG (commsys_stream_simulator): Frames encoded = "
            << frames_encoded << std::endl;
#endif
      }

   // Get access to the commsys object in stream-oriented mode
   commsys_stream<S>& s = dynamic_cast<commsys_stream<S>&> (*this->sys);
   // Demodulate -> Inverse Map -> Translate
   s.receive_path(received.extract(0, length), sof_prior, eof_prior, offset);
   // Store posterior end-of-frame drift probabilities
   eof_post = s.get_eof_post();
   // update counters
   frames_decoded++;

   // Determine estimated drift
   estimated_drift = libbase::index_of_max(eof_post) - offset;
   // Centralize posterior probabilities
   eof_post = 0;
   const int sh_a = std::max(0, -estimated_drift);
   const int sh_b = std::max(0, estimated_drift);
   const int sh_n = eof_post.size() - abs(estimated_drift);
   eof_post.segment(sh_a, sh_n) = s.get_eof_post().extract(sh_b, sh_n);
   // Tell user what we're doing
#if DEBUG>=3
   std::cerr << "DEBUG (commsys_stream_simulator): eof prior = " << eof_prior
         << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): eof post = " << eof_post
         << std::endl;
#endif

   // Determine actual cumulative drift and error in drift estimation
   assert(!actual_drift.empty());
   const int actual_drift_this = actual_drift.front();
   actual_drift.pop_front();
   drift_error += estimated_drift - actual_drift_this;
   // Tell user what we're doing
#if DEBUG>=2
   std::cerr << "DEBUG (commsys_stream_simulator): Actual frame drift = "
         << actual_drift_this << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Estimated frame drift = "
         << estimated_drift << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Acc. drift error at eof = "
         << drift_error << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Frames decoded = "
         << frames_decoded << std::endl;
#endif

   // Get source message to compare against
   assert(!source.empty());
   libbase::vector<int> source_this = source.front();
   source.pop_front();
   // Initialise result vector
   result.init(Base::count());
   result = 0;
   // For every iteration
   libbase::vector<int> decoded;
   for (int i = 0; i < this->sys->num_iter(); i++)
      {
      // Decode & update results
      this->sys->decode(decoded);
      R::updateresults(result, i, source_this, decoded);
      }
   // Keep record of what we last simulated
   this->last_event = concatenate(source_this, decoded);
   }

// Description & Serialization

template <class S, class R>
std::string commsys_stream_simulator<S, R>::description() const
   {
   std::ostringstream sout;
   sout << "Stream-oriented ";
   sout << Base::description();
   if (N > 0)
      sout << ", stream reset every " << N << " frames";
   return sout.str();
   }

// object serialization - saving

template <class S, class R>
std::ostream& commsys_stream_simulator<S, R>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Frame count between resets (0=don't reset)" << std::endl;
   sout << N << std::endl;
   // continue writing underlying system
   Base::serialize(sout);
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering; added frame count to reset
 */

template <class S, class R>
std::istream& commsys_stream_simulator<S, R>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // handle old-format files
   if (sin.fail())
      {
      version = 0;
      sin.clear();
      }
   // read frame count between resets
   N = 0;
   if (version >= 1)
      sin >> libbase::eatcomments >> N >> libbase::verify;
   // continue reading underlying system
   Base::serialize(sin);
   reset();
   assertalways(sin.good());
   return sin;
   }

} // end namespace

#include "gf.h"
#include "result_collector/commsys/errors_hamming.h"
#include "result_collector/commsys/errors_levenshtein.h"
#include "result_collector/commsys/prof_burst.h"
#include "result_collector/commsys/prof_pos.h"
#include "result_collector/commsys/prof_sym.h"
#include "result_collector/commsys/hist_symerr.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ
#define COLLECTOR_TYPE_SEQ \
   (errors_hamming) \
   (errors_levenshtein) \
   (prof_burst) \
   (prof_pos) \
   (prof_sym) \
   (hist_symerr)

/* Serialization string: commsys_stream_simulator<type,collector>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      collector = errors_hamming | errors_levenshtein | ...
 */
#define INSTANTIATE(r, args) \
      template class commsys_stream_simulator<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer commsys_stream_simulator<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "experiment", \
            "commsys_stream_simulator<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            commsys_stream_simulator<BOOST_PP_SEQ_ENUM(args)>::create); \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(COLLECTOR_TYPE_SEQ))

} // end namespace
