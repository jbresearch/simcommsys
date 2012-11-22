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

#include "tvb.h"
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

/*! \brief Determines and returns codebook to be used at index 'i'
 *
 * In determining the codebook to be used, this method advances the internal
 * state as necessary. This means that this method should only be called once
 * per index, in a deterministic order (ideally sequentially for all 'i').
 *
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
libbase::vector<libbase::vector<sig> > tvb<sig, real, real2>::select_codebook(
      const int i) const
   {
   // Initialize space for result
   array1vs_t codebook;
   libbase::allocate(codebook, q, n);
   // Select codebook
   int cb_index = 0;
   switch (codebook_type)
      {
      case codebook_sparse:
         assert(num_codebooks() == 1);
         cb_index = 0;
         break;

      case codebook_random:
         for (int d = 0; d < q; d++)
            for (bool ready = false; !ready;)
               {
               for (int s = 0; s < n; s++)
                  codebook(d)(s) = sig(this->r.ival(
                        field_utils<sig>::elements()));
               // this entry should be distinct from earlier entries
               ready = true;
               for (int dd = 0; dd < d; dd++)
                  if (codebook(dd).isequalto(codebook(d)))
                     {
                     ready = false;
                     break;
                     }
               }
         return codebook;

      case codebook_user_sequential:
         assert(num_codebooks() >= 1);
         cb_index = i % num_codebooks();
         break;

      case codebook_user_random:
         assert(num_codebooks() >= 1);
         cb_index = r.ival(num_codebooks());
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   // Copy chosen codebook and return
   for (int d = 0; d < q; d++)
      codebook(d) = codebook_tables(cb_index, d);
   return codebook;
   }

/*! \brief Determines and returns marker vector to be used at index 'i'
 *
 * In determining the marker to be used, this method advances the internal
 * state as necessary. This means that this method should only be called once
 * per index, in a deterministic order (ideally sequentially for all 'i').
 *
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
libbase::vector<sig> tvb<sig, real, real2>::select_marker(const int i) const
   {
   // Initialize space for result
   array1s_t marker;
   marker.init(n);
   // Select marker vector
   int mv_index = 0;
   switch (marker_type)
      {
      case marker_zero:
         marker = 0;
         return marker;

      case marker_random:
         for (int s = 0; s < n; s++)
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

/*! \brief Fills the entries of the encoding table, as requested
 *
 * This method calls select_codebook() and select_marker() sequentially for
 * each index requested, starting with zero and repeating 'length' times.
 * The result is placed in the corresponding index in encoding_table, offset
 * by the given amount.
 *
 * In determining the codebook and marker to be used at each index, this method
 * therefore advances the internal state as necessary. This means that this
 * method should only be called once for the given index range, in a
 * deterministic order over the range of indices for a frame (ideally
 * sequentially for all indices).
 *
 * \note The encoding table passed as a parameter must be already allocated
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
void tvb<sig, real, real2>::fill_encoding_table(array2vs_t& encoding_table,
      const int offset, const int length) const
   {
   // Sanity checks
   assert(offset >= 0 && offset < encoding_table.size().rows());
   assert(length > 0 && offset + length <= encoding_table.size().rows());
   assert(q == encoding_table.size().cols());
   assert(length <= this->input_block_size());
   // Advance and fill encoding table
   for (int i = 0; i < length; i++)
      {
      // Select codebook and marker vector
      const array1vs_t codebook = select_codebook(i);
      const array1s_t marker = select_marker(i);
#if DEBUG>=2
      std::cerr << "Codebook for i = " << i << std::endl;
      showcodebook(std::cerr, codebook);
      std::cerr << "Marker for i = " << i << std::endl;
      std::cerr << "\t";
      marker.serialize(std::cerr, ' ');
#endif
      // Encode each possible input symbol
      for (int d = 0; d < q; d++)
         encoding_table(offset + i, d) = field_utils<sig>::add(marker,
               codebook(d));
      }
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::advance() const
   {
   // Inherit sizes
   const int N = this->input_block_size();
   // Advance only for non-zero block sizes
   if (N > 0)
      {
      // Initialize space for encoding table
      libbase::allocate(encoding_table, N + lookahead, q, n);
      // Advance this system and set up the corresponding encoding table
      fill_encoding_table(encoding_table, 0, N);
      // if we have lookahead, make a copy and advance as necessary
      if (lookahead)
         {
         // make a copy of this object
         tvb<sig, real, real2> copy(*this);
         // advance the copy and set up the corresponding entries in table
         for (int offset = N, left = lookahead; left > 0; offset += N, left
               -= N)
            {
            const int length = std::min(N, left);
            copy.fill_encoding_table(encoding_table, offset, length);
            }
         }
      }
   // initialize our embedded metric computer
   fba.get_receiver().init(encoding_table);
   }

// encoding and decoding functions

template <class sig, class real, class real2>
void tvb<sig, real, real2>::domodulate(const int N, const array1i_t& encoded,
      array1s_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single codeword
   assertalways(N == q);
   // Initialise result vector
   tx.init(n * tau);
   assertalways(encoding_table.size().rows() == tau + lookahead);
   assertalways(encoding_table.size().cols() == q);
   // Encode source stream
   for (int i = 0; i < tau; i++)
      tx.segment(i * n, n) = encoding_table(i, encoded(i));
#if DEBUG>=3
   std::cerr << "encoded = " << encoded << std::endl;
   std::cerr << "tx = " << tx << std::endl;
#endif
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, array1vd_t& ptable)
   {
   const array1vd_t app; // empty APP table
   dodemodulate(chan, rx, app, ptable);
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   // Initialize for known-start
   init(chan);
   // Shorthand for transmitted and received frame sizes
   const int tau = this->output_block_size();
   const int rho = rx.size();
   // Algorithm parameters
   const int xmax = fba.get_xmax();
   // Check that rx size is within valid range
   assertalways(xmax >= abs(rho - tau));
   // Set up start-of-frame drift pdf (drift = 0)
   array1d_t sof_prior;
   sof_prior.init(2 * xmax + 1);
   sof_prior = 0;
   sof_prior(xmax + 0) = 1;
   // Set up end-of-frame drift pdf (drift = rho-tau)
   array1d_t eof_prior;
   eof_prior.init(2 * xmax + 1);
   eof_prior = 0;
   eof_prior(xmax + rho - tau) = 1;
   // Offset rx by xmax and pad to a total size of tau+2*xmax
   array1s_t r;
   r.init(tau + 2 * xmax);
   r.segment(xmax, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, 0, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(xmax));
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const libbase::size_type<libbase::vector> lookahead,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vd_t& ptable, array1d_t& sof_post,
      array1d_t& eof_post, const libbase::size_type<libbase::vector> offset)
   {
   // Initialize for given start distribution
   init(chan, sof_prior, offset);
   // TODO: validate priors have required size?
#ifndef NDEBUG
   std::cerr << "DEBUG (tvb): offset = " << offset << ", xmax = "
         << fba.get_xmax() << "." << std::endl;
#endif
   assert(offset == fba.get_xmax());
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
void tvb<sig, real, real2>::demodulate_wrapper(const channel<sig>& chan,
      const array1s_t& rx, const int lookahead, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Inherit block size from last modulation step
   const int N = this->input_block_size();
   // In cases with lookahead, extend app table if supplied
   array1vd_t app_x;
   if (lookahead > 0 && app.size() > 0)
      {
      // Initialise extended app table (one symbol per timestep)
      assert(lookahead % n == 0);
      libbase::allocate(app_x, N + lookahead / n, q);
      app_x = 1.0; // equiprobable
      // Copy supplied prior to initial segment
      assert(app.size() == N);
      app_x.segment(0, N) = app;
      }
   else
      app_x = app;
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
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba.decode(*this, rx, sof_prior, eof_prior, app_x, ptable_r, sof_post_r,
         eof_post_r, offset);
   // In cases with lookahead, re-compute EOF posterior at actual frame boundary
   if (lookahead > 0)
      fba.get_drift_pdf(eof_post_r, N);
   libbase::normalize_results(ptable_r.extract(0, N), ptable);
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
void tvb<sig, real, real2>::init(const channel<sig>& chan,
      const array1d_t& sof_pdf, const int offset)
   {
   // Inherit block size from last modulation step (and include lookahead)
   const int N = this->input_block_size() + lookahead;
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const qids<sig, real2>&> (chan);
   // Set channel block size to q-ary symbol size
   mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const int I = mychan.compute_I(tau);
   // No need to recompute xmax if we are given a prior PDF
   const int xmax = sof_pdf.size() > 0 ? offset : mychan.compute_xmax(tau,
         sof_pdf, offset);
   const int dxmax = mychan.compute_xmax(n);
   checkforchanges(I, xmax);
   // Determine whether to use global storage
   bool globalstore = false; // set to avoid compiler warning
   const int required = fba.get_memory_required(N, n, q, I, xmax, dxmax);
   switch (storage_type)
      {
      case storage_local:
         globalstore = false;
         break;

      case storage_global:
         globalstore = true;
         break;

      case storage_conditional:
         globalstore = (required <= globalstore_limit);
         checkforchanges(globalstore, required);
         break;

      default:
         failwith("Unknown storage mode");
         break;
      }
   // Initialize forward-backward algorithm
   fba.init(N, n, q, I, xmax, dxmax, th_inner, th_outer, flags.norm,
         flags.batch, flags.lazy, globalstore);
   // initialize our embedded metric computer with unchanging elements
   fba.get_receiver().init(n, mychan);
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::init()
   {
   // Build codebook if necessary
   switch (codebook_type)
      {
      case codebook_sparse:
         {
         // sparse codebooks defined only for GF(2)
         assertalways(field_utils<sig>::elements() == 2);
         libbase::sparse mycodebook(q, n);
         codebook_tables.init(1, q);
         for (int i = 0; i < q; i++)
            {
            const libbase::bitfield codeword(mycodebook(i), n);
            codebook_tables(0, i) = codeword.asvector();
            }
         }
         break;

      case codebook_random:
         // gets generated per symbol index on advance()
         codebook_tables.init(0, 0);
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         // nothing to do
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
#ifndef NDEBUG
   // If applicable, display codebook
   if (n > 2 && codebook_type != codebook_random)
      showcodebooks(libbase::trace);
#endif
   // If necessary, validate codebook
   if (codebook_type != codebook_random)
      validatecodebook();
   // Seed the generator and clear the per-frame encoding table
   r.seed(0);
   encoding_table.init(0, 0);
   // Check that everything makes sense
   test_invariant();
   }

/*!
 * \brief Check that all entries in table have correct length
 */
template <class sig, class real, class real2>
void tvb<sig, real, real2>::validate_sequence_length(const array1vs_t& table) const
   {
   assertalways(table.size() > 0);
   for (int i = 0; i < table.size(); i++)
      assertalways(table(i).size() == n);
   }

/*!
 * \brief Set up codebook with the given codewords
 */
template <class sig, class real, class real2>
void tvb<sig, real, real2>::copycodebook(const int i,
      const array1vs_t& codebook_s)
   {
   assertalways(codebook_s.size() == q);
   validate_sequence_length(codebook_s);
   // insert into matrix
   assertalways(i >= 0 && i < num_codebooks());
   assertalways(codebook_tables.size().cols() == q);
   codebook_tables.insertrow(codebook_s, i);
   }

/*!
 * \brief Display given codebook on given stream
 */

template <class sig, class real, class real2>
void tvb<sig, real, real2>::showcodebook(std::ostream& sout,
      const array1vs_t& codebook) const
   {
   assert(codebook.size() == q);
   for (int d = 0; d < q; d++)
      {
      sout << d << "\t";
      codebook(d).serialize(sout, ' ');
      }
   }

/*!
 * \brief Display codebook on given stream
 */

template <class sig, class real, class real2>
void tvb<sig, real, real2>::showcodebooks(std::ostream& sout) const
   {
   assert(num_codebooks() >= 1);
   assert(codebook_tables.size().cols() == q);
   for (int i = 0; i < num_codebooks(); i++)
      {
      sout << "Codebook " << i << ":" << std::endl;
      showcodebook(sout, codebook_tables.extractrow(i));
      }
   }

/*!
 * \brief Confirm that codebook is valid
 * Checks that all codebook entries are within range and that there are no
 * duplicate entries.
 */

template <class sig, class real, class real2>
void tvb<sig, real, real2>::validatecodebook() const
   {
   assertalways(num_codebooks() >= 1);
   assertalways(codebook_tables.size().cols() == q);
   for (int i = 0; i < num_codebooks(); i++)
      for (int d = 0; d < q; d++)
         {
         // all entries should be distinct for each codebook index
         for (int dd = 0; dd < d; dd++)
            assertalways(codebook_tables(i, dd).isnotequalto(codebook_tables(i, d)));
         //assertalways(codebook(i, dd) != codebook(i, d));
         }
   }

//! Inform user if I or xmax have changed (debug build only)

template <class sig, class real, class real2>
void tvb<sig, real, real2>::checkforchanges(int I, int xmax) const
   {
#ifndef NDEBUG
   static int last_I = 0;
   static int last_xmax = 0;
   if (last_I != I || last_xmax != xmax)
      {
      std::cerr << "TVB: I = " << I << ", xmax = " << xmax << std::endl;
      last_I = I;
      last_xmax = xmax;
      }
#endif
   }

//! Inform user if storage mode has changed

template <class sig, class real, class real2>
void tvb<sig, real, real2>::checkforchanges(bool globalstore, int required) const
   {
   static bool first_time = true;
   static bool last_globalstore = false;
   if (first_time || last_globalstore != globalstore)
      {
      std::cerr << "FBA Global Store ";
      if (globalstore)
         std::cerr << "Enabled";
      else
         std::cerr << "Disabled";
      std::cerr << ", Required: " << required << "MiB" << std::endl;
      last_globalstore = globalstore;
      first_time = false;
      }
   }

// Marker-specific setup functions

template <class sig, class real, class real2>
void tvb<sig, real, real2>::set_thresholds(const real th_inner,
      const real th_outer)
   {
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
   }

// description output

template <class sig, class real, class real2>
std::string tvb<sig, real, real2>::description() const
   {
   std::ostringstream sout;
   sout << "Time-Varying Block Code (" << n << "," << q << ", ";
   switch (codebook_type)
      {
      case codebook_sparse:
         sout << "sparse codebook";
         break;

      case codebook_random:
         sout << "random codebooks";
         break;

      case codebook_user_sequential:
         sout << codebook_name << " codebook ["
               << codebook_tables.size().rows() << ", sequential]";
         break;

      case codebook_user_random:
         sout << codebook_name << " codebook ["
               << codebook_tables.size().rows() << ", random]";
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   switch (marker_type)
      {
      case marker_zero:
         sout << ", no marker";
         break;

      case marker_random:
         sout << ", random marker";
         break;

      case marker_user_sequential:
         sout << ", AMVs [" << marker_vectors.size() << ", sequential]";
         break;

      case marker_user_random:
         sout << ", AMVs [" << marker_vectors.size() << ", random]";
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   sout << ", thresholds " << th_inner << "/" << th_outer;
   if (flags.norm)
      sout << ", normalized";
   if (flags.batch)
      sout << ", batch interface";
   else
      sout << ", single interface";
   if (flags.lazy)
      sout << ", lazy computation";
   else
      sout << ", pre-computation";
   switch (storage_type)
      {
      case storage_local:
         sout << ", local storage";
         break;

      case storage_global:
         sout << ", global storage";
         break;

      case storage_conditional:
         sout << ", global storage [≤" << globalstore_limit << " MiB]";
         break;

      default:
         failwith("Unknown storage mode");
         break;
      }
   if (lookahead == 0)
      sout << ", no look-ahead";
   else
      sout << ", look-ahead " << lookahead << " codewords";
   sout << "), " << fba.description();
   return sout.str();
   }

// object serialization - saving

template <class sig, class real, class real2>
std::ostream& tvb<sig, real, real2>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 6 << std::endl;
   sout << "#: Inner threshold" << std::endl;
   sout << th_inner << std::endl;
   sout << "#: Outer threshold" << std::endl;
   sout << th_outer << std::endl;
   sout << "# Normalize metrics between time-steps?" << std::endl;
   sout << flags.norm << std::endl;
   sout << "# Use batch receiver computation?" << std::endl;
   sout << flags.batch << std::endl;
   sout << "# Lazy computation of gamma?" << std::endl;
   sout << flags.lazy << std::endl;
   sout << "# Storage mode for gamma (0=local, 1=global, 2=conditional)" << std::endl;
   sout << storage_type << std::endl;
   if (storage_type == storage_conditional)
      {
      sout << "#: Memory threshold for global storage (in MiB)" << std::endl;
      sout << globalstore_limit << std::endl;
      }
   sout << "# Number of codewords to look ahead when stream decoding"
         << std::endl;
   sout << lookahead << std::endl;
   sout << "# n" << std::endl;
   sout << n << std::endl;
   sout << "# q" << std::endl;
   sout << q << std::endl;
   sout << "# codebook type (0=sparse, 1=random, 2=user[seq], 3=user[ran])"
         << std::endl;
   sout << codebook_type << std::endl;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         sout << "#: codebook name" << std::endl;
         sout << codebook_name << std::endl;
         sout << "#: codebook count" << std::endl;
         sout << num_codebooks() << std::endl;
         assert(num_codebooks() >= 1);
         assert(codebook_tables.size().cols() == q);
         for (int i = 0; i < num_codebooks(); i++)
            {
            sout << "#: codebook entries (table " << i << ")" << std::endl;
            for (int d = 0; d < q; d++)
               {
               codebook_tables(i, d).serialize(sout, ' ');
               //sout << std::endl;
               }
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   sout << "# marker type (0=zero, 1=random, 2=user[seq], 3=user[ran])"
         << std::endl;
   sout << marker_type << std::endl;
   switch (marker_type)
      {
      case marker_zero:
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
 * \version 1 Initial version - based on dminner v.2, except:
 *      - no user-threshold flag (specification is mandatory)
 *      - codebook and marker type field has incompatible values
 *      - codebook stored as vectors (so lsb is on left rather than right)
 *
 * \version 2 Added normalization, batch, lazy, caching flags; caching is only
 *      specified if lazy is true (otherwise it is meaningless)
 *
 * \version 3 Changed 'caching' flag to 'global store', now also defined for
 *      pre-computation cases
 *
 * \version 4 Added look-ahead quantity for stream decoding
 *
 * \version 5 Replaced globalstore flag with storage mode and memory threshold
 *
 * \version 6 Replaced k with q
 */

template <class sig, class real, class real2>
std::istream& tvb<sig, real, real2>::serialize(std::istream& sin)
   {
   int temp;
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // read thresholds
   sin >> libbase::eatcomments >> th_inner >> libbase::verify;
   sin >> libbase::eatcomments >> th_outer >> libbase::verify;
   // read decoder parameters
   if (version >= 2)
      {
      sin >> libbase::eatcomments >> flags.norm >> libbase::verify;
      sin >> libbase::eatcomments >> flags.batch >> libbase::verify;
      sin >> libbase::eatcomments >> flags.lazy >> libbase::verify;
      }
   else
      {
      flags.norm = true;
      flags.batch = true;
      flags.lazy = true;
      }
   // read storage mode
   if (version >= 5)
      {
      sin >> libbase::eatcomments >> temp >> libbase::verify;
      storage_type = (storage_t) temp;
      if (storage_type == storage_conditional)
         sin >> libbase::eatcomments >> globalstore_limit >> libbase::verify;
      }
   else if ((version >= 2 && flags.lazy) || version >= 3)
      {
      // read old globalstore flag
      sin >> libbase::eatcomments >> temp >> libbase::verify;
      storage_type = temp ? storage_global : storage_local;
      }
   else
      storage_type = storage_global;
   // read look-ahead quantity
   if (version >= 4)
      sin >> libbase::eatcomments >> lookahead >> libbase::verify;
   else
      lookahead = 0;
   // read code size
   sin >> libbase::eatcomments >> n >> libbase::verify;
   if (version >= 6)
      sin >> libbase::eatcomments >> q >> libbase::verify;
   else
      {
      int k;
      sin >> libbase::eatcomments >> k >> libbase::verify;
      q = int(pow(field_utils<sig>::elements(), k));
      }
   // read codebook
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   codebook_type = (codebook_t) temp;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         // gets generated automatically
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         sin >> libbase::eatcomments >> codebook_name >> libbase::verify;
         // read codebook count
         sin >> libbase::eatcomments >> temp >> libbase::verify;
         // allocate memory
         codebook_tables.init(temp, q);
         for (int i = 0; i < num_codebooks(); i++)
            {
            // read codebook from stream
            array1vs_t codebook_s;
            libbase::allocate(codebook_s, q, n);
            sin >> libbase::eatcomments;
            for (int d = 0; d < q; d++)
               {
               codebook_s(d).serialize(sin);
               libbase::verify(sin);
               }
            // copy read codebook
            copycodebook(i, codebook_s);
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   // read marker sequence type
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   marker_type = (marker_t) temp;
   switch (marker_type)
      {
      case marker_zero:
      case marker_random:
         // gets generated automatically
         break;

      case marker_user_sequential:
      case marker_user_random:
         // read count of modification vectors
         sin >> libbase::eatcomments >> temp >> libbase::verify;
         // read modification vectors from stream
         libbase::allocate(marker_vectors, temp, n);
         sin >> libbase::eatcomments;
         for (int i = 0; i < temp; i++)
            {
            marker_vectors(i).serialize(sin);
            libbase::verify(sin);
            }
         // validate list of modification vectors
         validate_sequence_length(marker_vectors);
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
#ifdef USE_CUDA
#define REAL_TYPE_SEQ \
   (float)(double)
#else
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)
#endif
#define REAL2_TYPE_SEQ \
   (float)(double)

/* Serialization string: tvb<type,real,real2>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | logrealfast (CPU only)
 *      real2 = float | double
 */
#define INSTANTIATE(r, args) \
      template class tvb<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer tvb<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "tvb<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            tvb<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (ALL_SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
