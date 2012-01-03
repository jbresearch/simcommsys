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
#include "result_collector/commsys_errors_levenshtein.h"

#include "commsys_stream.h"
#include "channel/bsid.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

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
   do
      {
      // Advance by one frame
      received_prev = received_this;
      source_this = source_next;
      received_this = received_next;
      // Create next source frame
      source_next = Base::createsource();
      // Encode -> Map -> Modulate next frame
      libbase::vector<S> transmitted = sys_tx->encode_path(source_next);
      // Transmit next frame
      received_next = sys_tx->transmit(transmitted);
#ifndef NDEBUG
      // update counters
      frames_encoded++;
#endif
      } while (source_this.size() == 0);

   // Shorthand for transmitted and received frame sizes
   const int tau = this->sys->output_block_size();
   const int rho = received_this.size();

   // Get access to the commsys channel object in stream-oriented mode
   const bsid& c = dynamic_cast<const bsid&> (*this->sys->getchan());
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
      // Initialize previous frame so we have something to copy
      received_prev.init(offset);
      received_prev = 0; // value not important as content is unused
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

   // Assemble stream
   libbase::vector<S> stream = concatenate(received_prev, received_this,
         received_next);
   // Extract received vector
   const int start = received_prev.size() - offset + drift_error;
   const int length = tau + eof_prior.size() - 1;
   assertalways(start >= 0 && start <= stream.size());
   assertalways(length >= 0 && length <= stream.size() - start);
   libbase::vector<S> received = stream.extract(start, length);

   // Get access to the commsys object in stream-oriented mode
   commsys_stream<S>& s = dynamic_cast<commsys_stream<S>&> (*this->sys);
   // Demodulate -> Inverse Map -> Translate
   s.receive_path(received, sof_prior, eof_prior, offset);
   // Store posterior end-of-frame drift probabilities
   eof_post = s.get_eof_post();
#ifndef NDEBUG
   // update counters
   frames_decoded++;
#endif

   // Determine estimated drift
   const int drift = libbase::index_of_max(eof_post) - offset;
   // Centralize posterior probabilities
   eof_post = 0;
   const int sh_a = std::max(0, -drift);
   const int sh_b = std::max(0, drift);
   const int sh_n = eof_post.size() - abs(drift);
   eof_post.segment(sh_a, sh_n) = s.get_eof_post().extract(sh_b, sh_n);
   // Determine actual cumulative drift and error in drift estimation
   cumulative_drift += rho - tau;
   drift_error += drift - (rho - tau);
#ifndef NDEBUG
   std::cerr << "DEBUG (commsys_stream_simulator): Actual acc. drift at sof = "
         << cumulative_drift - (rho - tau) << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Actual frame drift = "
         << rho - tau << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Actual acc. drift at eof = "
         << cumulative_drift << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Acc. drift error at sof = "
         << drift_error - (drift - (rho - tau)) << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Estimated frame drift = "
         << drift << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Acc. drift error at eof = "
         << drift_error << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Frames encoded = "
         << frames_encoded << std::endl;
   std::cerr << "DEBUG (commsys_stream_simulator): Frames decoded = "
         << frames_decoded << std::endl;
#endif

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
   return sout.str();
   }

template <class S, class R>
std::ostream& commsys_stream_simulator<S, R>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   return sout;
   }

template <class S, class R>
std::istream& commsys_stream_simulator<S, R>::serialize(std::istream& sin)
   {
   reset();
   Base::serialize(sin);
   return sin;
   }

// Explicit Realizations

using libbase::serializer;
//using libbase::gf;

// realizations for default results collector

template class commsys_stream_simulator<bool> ;
template <>
const serializer commsys_stream_simulator<bool>::shelper("experiment",
      "commsys_stream_simulator<bool>", commsys_stream_simulator<bool>::create);

// realizations for levenshtein results collector

template class commsys_simulator<bool, commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<bool, commsys_errors_levenshtein>::shelper(
            "experiment", "commsys_stream_simulator<bool,levenshtein>",
            commsys_stream_simulator<bool, commsys_errors_levenshtein>::create);

} // end namespace
