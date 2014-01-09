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

#include "marker.h"
#include "sparse.h"
#include "timer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show codebooks on frame advance
// 3 - Show transmitted sequence and the encoded message it represents
// 4 - Show prior and posterior sof/eof probabilities when decoding
// 5 - Show prior and posterior symbol probabilities when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*! \brief Determines and returns marker vector to be used at index 'i'
 *
 * In determining the marker to be used, this method advances the internal
 * state as necessary. This means that this method should only be called once
 * per index, in a deterministic order (ideally sequentially for all 'i').
 *
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
libbase::vector<sig> marker<sig, real, real2>::select_marker(const int i) const
   {
   // Initialize space for result
   array1s_t marker;
   marker.init(m);
   // Select marker vector
   int mv_index = 0;
   switch (marker_type)
      {
      case marker_random:
         for (int s = 0; s < m; s++)
            marker(s) = sig(this->r.ival(field_utils<sig>::elements()));
         return marker;

      case marker_user_sequential:
         assert(marker_vectors.size() >= 1);
         mv_index = i % marker_vectors.size();
         break;

      case marker_user_random:
         assert(marker_vectors.size() >= 1);
         mv_index = r.ival(marker_vectors.size());
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   // Copy chosen marker vector and return
   marker = marker_vectors(mv_index);
   return marker;
   }

/*! \brief Fills the entries of the per-frame marker sequence, as requested
 *
 * This method calls select_marker() sequentially for each index requested,
 * starting with zero and repeating 'length' times.
 * The result is placed in the corresponding index in frame_marker_sequence,
 * offset by the given amount.
 *
 * In determining the marker to be used at each index, this method therefore
 * advances the internal state as necessary. This means that this method should
 * only be called once for the given index range, in a deterministic order over
 * the range of indices for a frame (ideally sequentially for all indices).
 *
 * \note The per-frame marker sequence table passed as a parameter must be
 * already allocated
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
void marker<sig, real, real2>::fill_frame_marker_sequence(
      array1vs_t& frame_marker_sequence, const int offset,
      const int length) const
   {
   // Sanity checks
   assert(offset >= 0 && offset < frame_marker_sequence.size());
   assert(length > 0 && offset + length <= frame_marker_sequence.size());
   assert(length <= this->input_block_size() / d);
   // Advance and fill encoding table
   for (int i = 0; i < length; i++)
      {
      // Select marker vector
      const array1s_t marker = select_marker(i);
#if DEBUG>=2
      std::cerr << "Marker for i = " << i << " + " << offset << ":" << std::endl;
      std::cerr << "\t";
      marker.serialize(std::cerr, ' ');
#endif
      // Copy to result
      frame_marker_sequence(offset + i) = marker;
      }
   }

template <class sig, class real, class real2>
void marker<sig, real, real2>::advance() const
   {
   // Inherit sizes
   const int N = this->input_block_size() / d;
   // Advance only for non-zero block sizes
   if (N > 0)
      {
      // Initialize space for per-frame marker sequence
      libbase::allocate(frame_marker_sequence, N + lookahead, m);
      // Advance this system and set up the corresponding encoding table
      fill_frame_marker_sequence(frame_marker_sequence, 0, N);
      // if we have lookahead, make a copy and advance as necessary
      if (lookahead)
         {
         // make a copy of this object
         marker<sig, real, real2> copy(*this);
         // advance the copy and set up the corresponding entries in table
         for (int offset = N, left = lookahead; left > 0;
               offset += N, left -= N)
            {
            const int length = std::min(N, left);
            copy.fill_frame_marker_sequence(frame_marker_sequence, offset, length);
            }
         }
      }
   }

// encoding and decoding functions

template <class sig, class real, class real2>
void marker<sig, real, real2>::domodulate(const int q, const array1i_t& encoded,
      array1s_t& tx)
   {
   // Inherit sizes
   const int N = this->input_block_size() / d;
   // Check validity
   assertalways(N * d == this->input_block_size());
   assertalways(N * d == encoded.size());
   assertalways(q == field_utils<sig>::elements());
   assertalways(frame_marker_sequence.size() == N + lookahead);
   // Initialise result vector
   tx.init(N * (d + m));
   // Encode source stream
   for (int i = 0; i < N; i++)
      {
      // data bits first
      tx.segment(i * (d + m), d) = encoded.extract(i * d, d);
      // marker bits next
      tx.segment(i * (d + m) + d, m) = frame_marker_sequence(i);
      }
#if DEBUG>=3
   std::cerr << "encoded = " << encoded << std::endl;
   std::cerr << "tx = " << tx << std::endl;
#endif
   }

template <class sig, class real, class real2>
void marker<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, array1vd_t& ptable)
   {
   const array1vd_t app; // empty APP table
   dodemodulate(chan, rx, app, ptable);
   }

template <class sig, class real, class real2>
void marker<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   // Initialize for known-start
   init(chan);
   // Shorthand for transmitted and received frame sizes
   const int tau = this->output_block_size();
   const int rho = rx.size();
   // Algorithm parameters
   const int mtau_min = fba.get_mtau_min();
   const int mtau_max = fba.get_mtau_max();
   // Check that rx size is within valid range
   assertalways(mtau_max >= abs(rho - tau));
   // Set up start-of-frame drift pdf (drift = 0)
   array1d_t sof_prior;
   sof_prior.init(mtau_max - mtau_min + 1);
   sof_prior = 0;
   sof_prior(0 - mtau_min) = 1;
   // Set up end-of-frame drift pdf (drift = rho-tau)
   array1d_t eof_prior;
   eof_prior.init(mtau_max - mtau_min + 1);
   eof_prior = 0;
   eof_prior(rho - tau - mtau_min) = 1;
   // Offset rx by mtau_max and pad to a total size of tau+mtau_max-mtau_min
   array1s_t r;
   r.init(tau + mtau_max - mtau_min);
   r.segment(mtau_max, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, 0, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(mtau_max));
   }

template <class sig, class real, class real2>
void marker<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const libbase::size_type<libbase::vector> lookahead,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
      array1d_t& eof_post, const libbase::size_type<libbase::vector> offset)
   {
   // Initialize for given start distribution
   init(chan, sof_prior, offset);
   // TODO: validate priors have required size?
#ifndef NDEBUG
   std::cerr << "DEBUG (marker): offset = " << offset << ", mtau_min = "
         << fba.get_mtau_min() << "." << std::endl;
#endif
   assert(offset == -fba.get_mtau_min());
   // Delegate
   demodulate_wrapper(chan, rx, lookahead, sof_prior, eof_prior, app, ptable,
         sof_post, eof_post, offset);
   }

/*!
 * \brief Wrapper for calling demodulation algorithm
 *
 * This method assumes that the init() method has already been called with
 * the appropriate parameters.
 */
template <class sig, class real, class real2>
void marker<sig, real, real2>::demodulate_wrapper(const channel<sig>& chan,
      const array1s_t& rx, const int lookahead, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Inherit block size from last modulation step, and determine lookahead
   const int N = this->input_block_size() / d;
   const int L = lookahead / (d + m);
   assert(N * d == this->input_block_size());
   assert(L * (d + m) == lookahead);
   // Set up prior table for FBA
   array1vd_t app_x;
   libbase::allocate(app_x, (N + L) * (d + m), field_utils<sig>::elements());
   // Initialize to equiprobable
   app_x = 1.0 / field_utils<sig>::elements();
   // Copy priors for data bits, if supplied
   if (app.size() > 0)
      {
      assert(app.size() == N * d);
      for (int i = 0; i < N; i++)
         app_x.segment(i * (d + m), d) = app.extract(i * d, d);
      }
   // Determine priors for marker bits
   assert(frame_marker_sequence.size() == N + L);
   for (int i = 0; i < N + L; i++)
      {
      // Initialize priors for marker bits of codeword i
      app_x.segment(i * (d + m) + d, m) = 0.0;
      for (int j = 0; j < m; j++)
         app_x(i * (d + m) + d + j)(frame_marker_sequence(i)(j)) = 1.0;
      }
   // Call FBA and normalize results
#if DEBUG>=4
   using libbase::index_of_max;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "max at " << index_of_max(sof_prior) - offset << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
   std::cerr << "max at " << index_of_max(eof_prior) - offset << std::endl;
#endif
#if DEBUG>=5
   std::cerr << "app = " << app << std::endl;
   std::cerr << "app_x = " << app_x << std::endl;
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba.decode(*this, rx, sof_prior, eof_prior, app_x, ptable_r, sof_post_r,
         eof_post_r, offset);
   // Extract posteriors for data bits
   libbase::allocate(ptable, N * d, field_utils<sig>::elements());
   for (int i = 0; i < N; i++)
      for (int j = 0; j < d; j++)
         libbase::normalize(ptable_r(i * (d + m) + j), ptable(i * d + j));
   // In cases with lookahead, re-compute EOF posterior at actual frame boundary
   if (lookahead > 0)
      fba.get_drift_pdf(eof_post_r, N * (d + m));
   libbase::normalize(sof_post_r, sof_post);
   libbase::normalize(eof_post_r, eof_post);
#if DEBUG>=4
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "max at " << index_of_max(sof_post) - offset << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
   std::cerr << "max at " << index_of_max(eof_post) - offset << std::endl;
#endif
#if DEBUG>=5
   std::cerr << "ptable = " << ptable << std::endl;
#endif
   }

// Setup procedure

template <class sig, class real, class real2>
void marker<sig, real, real2>::init(const channel<sig>& chan,
      const array1d_t& sof_pdf, const int offset)
   {
   // Inherit block size from last modulation step (and include lookahead)
   const int N = this->input_block_size() / d + lookahead;
   const int tau = N * (d + m);
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const qids<sig, real2>&>(chan);
   // Set channel block size to actual block size
   mychan.set_blocksize(tau);
   // Set the probability of channel event outside chosen limits
   mychan.set_pr(Pr);
   // Determine required FBA parameter values
   // No need to recompute mtau_min/max if we are given a prior PDF
   int mtau_min = -offset;
   int mtau_max = sof_pdf.size() - offset - 1;
   if (sof_pdf.size() == 0)
      mychan.compute_limits(tau, Pr, mtau_min, mtau_max, sof_pdf, offset);
   int m1_min, m1_max;
   mychan.compute_limits(1, qids_utils::divide_error_probability(Pr, tau),
         m1_min, m1_max);
   checkforchanges(m1_min, m1_max, mtau_min, mtau_max);
   // Initialize forward-backward algorithm
   fba.init(tau, mtau_min, mtau_max, m1_min, m1_max, norm, mychan);
   }

template <class sig, class real, class real2>
void marker<sig, real, real2>::init()
   {
   // Seed the generator and clear the per-frame sequence of markers
   r.seed(0);
   frame_marker_sequence.init(0);
   // Check that everything makes sense
   test_invariant();
   }

/*!
 * \brief Check that all entries in table have correct length
 */
template <class sig, class real, class real2>
void marker<sig, real, real2>::validate_marker_length(
      const array1vs_t& table) const
   {
   assertalways(table.size() > 0);
   for (int i = 0; i < table.size(); i++)
      assertalways(table(i).size() == m);
   }

//! Inform user if state space limits have changed (debug build only)

template <class sig, class real, class real2>
void marker<sig, real, real2>::checkforchanges(int m1_min, int m1_max, int mtau_min, int mtau_max) const
   {
#ifndef NDEBUG
   static int last_m1_min = 0;
   static int last_m1_max = 0;
   if (last_m1_min != m1_min || last_m1_max != m1_max)
      {
      std::cerr << "DEBUG (marker): m1_min = " << m1_min << ", m1_max = " << m1_max << std::endl;
      last_m1_min = m1_min;
      last_m1_max = m1_max;
      }
   static int last_mtau_min = 0;
   static int last_mtau_max = 0;
   if (last_mtau_min != mtau_min || last_mtau_max != mtau_max)
      {
      std::cerr << "DEBUG (marker): mtau_min = " << mtau_min << ", mtau_max = " << mtau_max << std::endl;
      last_mtau_min = mtau_min;
      last_mtau_max = mtau_max;
      }
#endif
   }

// description output

template <class sig, class real, class real2>
std::string marker<sig, real, real2>::description() const
   {
   std::ostringstream sout;
   sout << "Marker Code (" << d << "," << m;
   switch (marker_type)
      {
      case marker_random:
         sout << ", random marker";
         break;

      case marker_user_sequential:
         sout << ", user [" << marker_vectors.size() << ", sequential]";
         break;

      case marker_user_random:
         sout << ", user [" << marker_vectors.size() << ", random]";
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   sout << ", Pr=" << Pr;
   if (norm)
      sout << ", normalized";
   if (lookahead == 0)
      sout << ", no look-ahead";
   else
      sout << ", look-ahead " << lookahead << " codewords";
   sout << "), " << fba.description();
   return sout.str();
   }

// object serialization - saving

template <class sig, class real, class real2>
std::ostream& marker<sig, real, real2>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 2 << std::endl;
   sout << "# Probability of channel event outside chosen limits" << std::endl;
   sout << Pr << std::endl;
   sout << "# Normalize metrics between time-steps?" << std::endl;
   sout << norm << std::endl;
   sout << "# Number of codewords to look ahead when stream decoding"
         << std::endl;
   sout << lookahead << std::endl;
   sout << "# d (number of data symbols between markers)" << std::endl;
   sout << d << std::endl;
   sout << "# m (length of marker sequence)" << std::endl;
   sout << m << std::endl;
   sout << "# marker type (0=random, 1=user[seq], 2=user[ran])" << std::endl;
   sout << marker_type << std::endl;
   switch (marker_type)
      {
      case marker_random:
         break;

      case marker_user_sequential:
      case marker_user_random:
         sout << "#: marker vectors" << std::endl;
         sout << marker_vectors.size() << std::endl;
         for (int i = 0; i < marker_vectors.size(); i++)
            {
            marker_vectors(i).serialize(sout, ' ');
            //sout << std::endl;
            }
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 *
 * \version 2 Added probability of channel event outside chosen limits
 */

template <class sig, class real, class real2>
std::istream& marker<sig, real, real2>::serialize(std::istream& sin)
   {
   int temp;
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // read probability of channel event outside chosen limits
   if (version >= 2)
      sin >> libbase::eatcomments >> Pr >> libbase::verify;
   else
      Pr = 1e-10;
   // read decoder parameters
   sin >> libbase::eatcomments >> norm >> libbase::verify;
   // read look-ahead quantity
   sin >> libbase::eatcomments >> lookahead >> libbase::verify;
   // read code size
   sin >> libbase::eatcomments >> d >> libbase::verify;
   sin >> libbase::eatcomments >> m >> libbase::verify;
   // read marker sequence type
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   marker_type = (marker_t) temp;
   switch (marker_type)
      {
      case marker_random:
         // gets generated automatically
         break;

      case marker_user_sequential:
      case marker_user_random:
         // read count of marker vectors
         sin >> libbase::eatcomments >> temp >> libbase::verify;
         // read marker vectors from stream
         libbase::allocate(marker_vectors, temp, m);
         sin >> libbase::eatcomments;
         for (int i = 0; i < temp; i++)
            {
            marker_vectors(i).serialize(sin);
            libbase::verify(sin);
            }
         // validate list of marker vectors
         validate_marker_length(marker_vectors);
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   init();
   return sin;
   }

} // end namespace

#include "gf.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define ALL_SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double)

/* Serialization string: marker<type,real,real2>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | logrealfast
 *      real2 = float | double
 */
#define INSTANTIATE(r, args) \
      template class marker<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer marker<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "marker<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            marker<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (ALL_SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
