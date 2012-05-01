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
#include "result_collector/commsys_errors_levenshtein.h"
#include "result_collector/commsys_prof_burst.h"
#include "result_collector/commsys_prof_pos.h"
#include "result_collector/commsys_prof_sym.h"
#include "result_collector/commsys_hist_symerr.h"

namespace libcomm {

// Explicit Realizations

using libbase::serializer;
using libbase::gf;

// realizations for default results collector

template class commsys_stream_simulator<sigspace> ;
template <>
const serializer commsys_stream_simulator<sigspace>::shelper("experiment",
      "commsys_stream_simulator<sigspace>",
      commsys_stream_simulator<sigspace>::create);

template class commsys_stream_simulator<bool> ;
template <>
const serializer commsys_stream_simulator<bool>::shelper("experiment",
      "commsys_stream_simulator<bool>", commsys_stream_simulator<bool>::create);

template class commsys_stream_simulator<gf<1, 0x3> > ;
template <>
const serializer commsys_stream_simulator<gf<1, 0x3> >::shelper("experiment",
      "commsys_stream_simulator<gf<1,0x3>>", commsys_stream_simulator<
            gf<1, 0x3> >::create);

template class commsys_stream_simulator<gf<2, 0x7> > ;
template <>
const serializer commsys_stream_simulator<gf<2, 0x7> >::shelper("experiment",
      "commsys_stream_simulator<gf<2,0x7>>", commsys_stream_simulator<
            gf<2, 0x7> >::create);

template class commsys_stream_simulator<gf<3, 0xB> > ;
template <>
const serializer commsys_stream_simulator<gf<3, 0xB> >::shelper("experiment",
      "commsys_stream_simulator<gf<3,0xB>>", commsys_stream_simulator<
            gf<3, 0xB> >::create);

template class commsys_stream_simulator<gf<4, 0x13> > ;
template <>
const serializer commsys_stream_simulator<gf<4, 0x13> >::shelper("experiment",
      "commsys_stream_simulator<gf<4,0x13>>", commsys_stream_simulator<gf<4,
            0x13> >::create);

template class commsys_stream_simulator<gf<5, 0x25> > ;
template <>
const serializer commsys_stream_simulator<gf<5, 0x25> >::shelper("experiment",
      "commsys_stream_simulator<gf<5,0x25>>", commsys_stream_simulator<gf<5,
            0x25> >::create);

template class commsys_stream_simulator<gf<6, 0x43> > ;
template <>
const serializer commsys_stream_simulator<gf<6, 0x43> >::shelper("experiment",
      "commsys_stream_simulator<gf<6,0x43>>", commsys_stream_simulator<gf<6,
            0x43> >::create);

template class commsys_stream_simulator<gf<7, 0x89> > ;
template <>
const serializer commsys_stream_simulator<gf<7, 0x89> >::shelper("experiment",
      "commsys_stream_simulator<gf<7,0x89>>", commsys_stream_simulator<gf<7,
            0x89> >::create);

template class commsys_stream_simulator<gf<8, 0x11D> > ;
template <>
const serializer commsys_stream_simulator<gf<8, 0x11D> >::shelper("experiment",
      "commsys_stream_simulator<gf<8,0x11D>>", commsys_stream_simulator<gf<8,
            0x11D> >::create);

template class commsys_stream_simulator<gf<9, 0x211> > ;
template <>
const serializer commsys_stream_simulator<gf<9, 0x211> >::shelper("experiment",
      "commsys_stream_simulator<gf<9,0x211>>", commsys_stream_simulator<gf<9,
            0x211> >::create);

template class commsys_stream_simulator<gf<10, 0x409> > ;
template <>
const serializer commsys_stream_simulator<gf<10, 0x409> >::shelper(
      "experiment", "commsys_stream_simulator<gf<10,0x409>>",
      commsys_stream_simulator<gf<10, 0x409> >::create);

// realizations for levenshtein results collector

template class commsys_stream_simulator<bool, commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<bool, commsys_errors_levenshtein>::shelper(
            "experiment", "commsys_stream_simulator<bool,levenshtein>",
            commsys_stream_simulator<bool, commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<1, 0x3> , commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<1, 0x3> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<1,0x3>,levenshtein>",
            commsys_stream_simulator<gf<1, 0x3> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<2, 0x7> , commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<2, 0x7> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<2,0x7>,levenshtein>",
            commsys_stream_simulator<gf<2, 0x7> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<3, 0xB> , commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<3, 0xB> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<3,0xB>,levenshtein>",
            commsys_stream_simulator<gf<3, 0xB> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<4, 0x13> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<4, 0x13> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<4,0x13>,levenshtein>",
            commsys_stream_simulator<gf<4, 0x13> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<5, 0x25> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<5, 0x25> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<5,0x25>,levenshtein>",
            commsys_stream_simulator<gf<5, 0x25> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<6, 0x43> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<6, 0x43> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<6,0x43>,levenshtein>",
            commsys_stream_simulator<gf<6, 0x43> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<7, 0x89> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<7, 0x89> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<7,0x89>,levenshtein>",
            commsys_stream_simulator<gf<7, 0x89> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<8, 0x11D> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<8, 0x11D> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<8,0x11D>,levenshtein>",
            commsys_stream_simulator<gf<8, 0x11D> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<9, 0x211> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<9, 0x211> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<9,0x211>,levenshtein>",
            commsys_stream_simulator<gf<9, 0x211> , commsys_errors_levenshtein>::create);

template class commsys_stream_simulator<gf<10, 0x409> ,
      commsys_errors_levenshtein> ;
template <>
const serializer
      commsys_stream_simulator<gf<10, 0x409> , commsys_errors_levenshtein>::shelper(
            "experiment",
            "commsys_stream_simulator<gf<10,0x409>,levenshtein>",
            commsys_stream_simulator<gf<10, 0x409> , commsys_errors_levenshtein>::create);

// realizations for non-default containers

// template class commsys_stream_simulator<bool,matrix>;
// template <>
// const serializer commsys_stream_simulator<bool,matrix>::shelper("experiment", "commsys_stream_simulator<bool,matrix>", commsys_stream_simulator<bool,matrix>::create);

// realizations for non-default results collectors

template class commsys_stream_simulator<bool, commsys_prof_burst> ;
template <>
const serializer commsys_stream_simulator<bool, commsys_prof_burst>::shelper(
      "experiment", "commsys_stream_simulator<bool,prof_burst>",
      commsys_stream_simulator<bool, commsys_prof_burst>::create);

template class commsys_stream_simulator<bool, commsys_prof_pos> ;
template <>
const serializer commsys_stream_simulator<bool, commsys_prof_pos>::shelper(
      "experiment", "commsys_stream_simulator<bool,prof_pos>",
      commsys_stream_simulator<bool, commsys_prof_pos>::create);

template class commsys_stream_simulator<bool, commsys_prof_sym> ;
template <>
const serializer commsys_stream_simulator<bool, commsys_prof_sym>::shelper(
      "experiment", "commsys_stream_simulator<bool,prof_sym>",
      commsys_stream_simulator<bool, commsys_prof_sym>::create);

template class commsys_stream_simulator<bool, commsys_hist_symerr> ;
template <>
const serializer commsys_stream_simulator<bool, commsys_hist_symerr>::shelper(
      "experiment", "commsys_stream_simulator<bool,hist_symerr>",
      commsys_stream_simulator<bool, commsys_hist_symerr>::create);

} // end namespace
