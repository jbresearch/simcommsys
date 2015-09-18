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

#include "tvb.h"
#include "algorithm/fba2-factory.h"
#include "sparse.h"
#include "timer.h"
#include "cputimer.h"
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

/*! \brief Determines and returns codebook to be used at index 'i'.
 * \return Index for codebook to use (into the codebook tables)
 *
 * In determining the codebook to be used, this method advances the internal
 * state as necessary. This means that this method should only be called once
 * per index, in a deterministic order (ideally sequentially for all 'i').
 *
 * \warning Although this is a const method, the internal state is updated
 */
template <class sig, class real, class real2>
int tvb<sig, real, real2>::select_codebook(const int i) const
   {
   // Select codebook
   int cb_index = 0;
   switch (codebook_type)
      {
      case codebook_sparse:
         assert(num_codebooks() == 1);
         cb_index = 0;
         break;

      case codebook_random:
         // NOTE: in the following, we cast away the const nature of
         //       codebook_tables; this won't matter because this is not saved
         //       if we have a random codebook.
         // Generate codebook
         {
         const int n = codebook_tables(0, 0).size();
         for (int d = 0; d < q; d++)
            for (bool ready = false; !ready;)
               {
               for (int s = 0; s < n; s++)
                  const_cast<sig&>(codebook_tables(0, d)(s)) = sig(
                        this->r.ival(field_utils<sig>::elements()));
               // this entry should be distinct from earlier entries
               ready = true;
               for (int dd = 0; dd < d; dd++)
                  if (codebook_tables(0, dd).isequalto(codebook_tables(0, d)))
                     {
                     ready = false;
                     break;
                     }
               }
         }
         break;

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
   // Return index to chosen codebook
   return cb_index;
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
libbase::vector<sig> tvb<sig, real, real2>::select_marker(const int i, const int n) const
   {
   array1s_t marker_vector;
   // Apply marker vector
   switch (marker_type)
      {
      case marker_zero:
         // nothing to do
         break;

      case marker_random:
         for (int s = 0; s < n; s++)
            marker_vector(s) = sig(this->r.ival(field_utils<sig>::elements()));
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   return marker_vector;
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
      const int cb_index = select_codebook(i);
      const int n = codebook_tables(cb_index, 0).size();
      const array1s_t marker_vector = select_marker(i, n);
#if DEBUG>=2
      std::cerr << "Codebook for i = " << i << std::endl;
      showcodebook(std::cerr, codebook_tables.row(cb_index));
      std::cerr << "Marker for i = " << i << std::endl;
      std::cerr << "\t";
      marker_vector.serialize(std::cerr, ' ');
#endif
      // Encode each possible input symbol
      for (int d = 0; d < q; d++)
         {
         encoding_table(offset + i, d) = codebook_tables(cb_index, d);
         // apply marker vector if necessary
         if (marker_vector.size() > 0)
            field_utils<sig>::add_to(encoding_table(offset + i, d),
                  marker_vector);
         }
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
      libbase::cputimer t("t_advance");
      // Initialize space for encoding table
      encoding_table.init(N + lookahead, q);
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
      const_cast<tvb<sig, real, real2>*>(this)->add_timer(t);
      }
   // indicate the encoding table has changed
   changed_encoding_table = true;
   }

// encoding and decoding functions

template <class sig, class real, class real2>
void tvb<sig, real, real2>::domodulate(const int q, const array1i_t& encoded,
      array1s_t& tx)
   {
   // Inherit sizes
   const int N = this->input_block_size();
   // Check validity
   assertalways(N == encoded.size());
   // Each 'encoded' symbol must be representable by a single codeword
   assertalways(this->q == q);
   // Determine length of frame
   int tau = 0;
   for (int i = 0; i < N; i++)
      tau += encoding_table(i, 0).size();
   // Initialise result vector
   tx.init(tau);
   assertalways(encoding_table.size().rows() == N + lookahead);
   assertalways(encoding_table.size().cols() == q);
   // Encode source stream
   for (int i = 0, j = 0; i < N; i++)
      {
      const int n = encoding_table(i, 0).size();
      tx.segment(j, n) = encoding_table(i, encoded(i));
      j += n;
      }
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
   // Offset rx by -mtau_min and pad to a total size of tau+mtau_max-mtau_min
   array1s_t r;
   r.init(tau + mtau_max - mtau_min);
   r.segment(-mtau_min, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, 0, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(-mtau_min));
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
   std::cerr << "DEBUG (tvb): offset = " << offset << ", mtau_min = "
         << mtau_min << "." << std::endl;
#endif
   assert(offset == -mtau_min);
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
   libbase::cputimer t1("t_priors");
   array1vd_t app_x;
   if (lookahead > 0 && app.size() > 0)
      {
      // Initialise extended app table (one symbol per timestep)
      const int n = get_avg_codeword_length();
      assert(lookahead % n == 0);
      libbase::allocate(app_x, N + lookahead / n, q);
      app_x = 1.0; // equiprobable
      // Copy supplied prior to initial segment
      assert(app.size() == N);
      app_x.segment(0, N) = app;
      }
   else
      app_x = app;
   this->add_timer(t1);
   // Initialize FBA metric computer as needed
   if (changed_encoding_table)
      {
      libbase::cputimer te("t_enctable");
      fba_ptr->init(encoding_table);
      changed_encoding_table = false;
      this->add_timer(te);
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
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba_ptr->decode(*this, rx, sof_prior, eof_prior, app_x, ptable_r, sof_post_r,
         eof_post_r, offset);
   // In cases with lookahead, re-compute EOF posterior at actual frame boundary
   libbase::cputimer t2("t_posteriors");
   if (lookahead > 0)
      fba_ptr->get_drift_pdf(eof_post_r, N);
   libbase::normalize_results(ptable_r.extract(0, N), ptable);
   libbase::normalize(sof_post_r, sof_post);
   libbase::normalize(eof_post_r, eof_post);
   this->add_timer(t2);
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
#ifndef NDEBUG
   libbase::cputimer t("t_init");
#endif
   // Inherit block size from last modulation step (and include lookahead)
   const int N = this->input_block_size() + lookahead;
   assert(N > 0);
   // Determine average frame length (including lookahead section)
   const int tau = get_sequence_length(this->input_block_size())
         + get_sequence_length(lookahead);
   // Determine longest codeword we need to work with
   const int nmax = get_max_codeword_length();
   // Copy channel for access within R()
   mychan.reset(dynamic_cast<channel_insdel<sig, real2>*> (chan.clone()));
   // Set channel block size to longest codeword
   mychan->set_blocksize(nmax);
   // Set the probability of channel event outside chosen limits
   mychan->set_pr(qids_utils::divide_error_probability(Pr, N));
   // Determine required FBA parameter values
   if (sof_pdf.size())
      {
      // No need to recompute mtau_min/max if we are given a prior PDF
      mtau_min = -offset;
      mtau_max = sof_pdf.size() - offset - 1;
      }
   else
      mychan->compute_limits(tau, Pr, mtau_min, mtau_max, sof_pdf, offset);
   int mn_min, mn_max;
   mychan->compute_limits(nmax, qids_utils::divide_error_probability(Pr, N),
         mn_min, mn_max);
   int m1_min, m1_max;
   mychan->compute_limits(1, qids_utils::divide_error_probability(Pr, tau),
         m1_min, m1_max);
   checkforchanges(m1_min, m1_max, mn_min, mn_max, mtau_min, mtau_max);
   //! Determine whether to use global storage
   static bool globalstore = false; // set to avoid compiler warning
   bool last_globalstore = globalstore; // keep track of last setting
   const int required = fba_type::get_memory_required(N, q, mtau_min, mtau_max,
         mn_min, mn_max);
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
   // Create an embedded algorithm object of the correct type, as needed
   if (!fba_ptr || globalstore != last_globalstore)
      {
      const bool fss = mychan->is_statespace_fixed();
      const bool thresholding = th_inner > real(0) || th_outer > real(0);
      fba_ptr = fba2_factory<sig, real, real2>::get_instance(fss, thresholding,
            flags.lazy, globalstore);
      // Mark the encoding table as changed, to force receiver init
      changed_encoding_table = true;
      }
   // Initialize forward-backward algorithm
   fba_ptr->init(N, q, mtau_min, mtau_max, mn_min, mn_max, m1_min, m1_max,
         th_inner, th_outer, mychan->get_computer());
#ifndef NDEBUG
   this->add_timer(t);
#endif
   }

template <class sig, class real, class real2>
void tvb<sig, real, real2>::init()
   {
#ifndef NDEBUG
   // If applicable, display codebook
   if (codebook_type != codebook_random)
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
 * \brief Check that all entries in table have the given length
 */
template <class sig, class real, class real2>
void tvb<sig, real, real2>::validate_sequence_length(const array1vs_t& table, const int n) const
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
   const int n = codebook_s(0).size();
   validate_sequence_length(codebook_s, n);
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
         }
   }

/*!
 * \brief Determine the average codeword length for given codebooks
 */

template <class sig, class real, class real2>
double tvb<sig, real, real2>::get_avg_codeword_length() const
   {
   double n = 0;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         assert(num_codebooks() == 1);
         n = codebook_tables(0, 0).size();
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         assert(num_codebooks() >= 1);
         for (int i = 0; i < num_codebooks(); i++)
            n += codebook_tables(i, 0).size();
         n /= num_codebooks();
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   return n;
   }

/*!
 * \brief Determine the longest codeword for given codebooks
 */

template <class sig, class real, class real2>
int tvb<sig, real, real2>::get_max_codeword_length() const
   {
   int n = 0;
   assert(num_codebooks() >= 1);
   for (int i = 0; i < num_codebooks(); i++)
      n = std::max(n, int(codebook_tables(i, 0).size()));
   return n;
   }

/*!
 * \brief Determine the codeword length (single codebook only)
 */

template <class sig, class real, class real2>
int tvb<sig, real, real2>::get_codeword_length() const
   {
   assert(num_codebooks() == 1);
   return codebook_tables(0, 0).size();
   }

/*!
 * \brief Determine the frame length for the given number of symbols
 */

template <class sig, class real, class real2>
int tvb<sig, real, real2>::get_sequence_length(const int N) const
   {
   int tau = 0;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
      case codebook_user_random:
         tau = int(N * get_avg_codeword_length());
         break;

      case codebook_user_sequential:
         {
         const int blk = N / num_codebooks();
         tau = int(blk * num_codebooks() * get_avg_codeword_length());
         const int k = N % num_codebooks();
         for (int i = 0; i < k; i++)
            tau += codebook_tables(i, 0).size();
         }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   return tau;
   }

//! Inform user if state space limits have changed (debug build only)

template <class sig, class real, class real2>
void tvb<sig, real, real2>::checkforchanges(int m1_min, int m1_max, int mn_min,
      int mn_max, int mtau_min, int mtau_max) const
   {
#ifndef NDEBUG
   static int last_m1_min = 0;
   static int last_m1_max = 0;
   if (last_m1_min != m1_min || last_m1_max != m1_max)
      {
      std::cerr << "DEBUG (tvb): m1_min = " << m1_min << ", m1_max = " << m1_max << std::endl;
      last_m1_min = m1_min;
      last_m1_max = m1_max;
      }
   static int last_mn_min = 0;
   static int last_mn_max = 0;
   if (last_mn_min != mn_min || last_mn_max != mn_max)
      {
      std::cerr << "DEBUG (tvb): mn_min = " << mn_min << ", mn_max = " << mn_max << std::endl;
      last_mn_min = mn_min;
      last_mn_max = mn_max;
      }
   static int last_mtau_min = 0;
   static int last_mtau_max = 0;
   if (last_mtau_min != mtau_min || last_mtau_max != mtau_max)
      {
      std::cerr << "DEBUG (tvb): mtau_min = " << mtau_min << ", mtau_max = " << mtau_max << std::endl;
      last_mtau_min = mtau_min;
      last_mtau_max = mtau_max;
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
   this->th_inner = th_inner;
   this->th_outer = th_outer;
   test_invariant();
   }

// description output

template <class sig, class real, class real2>
std::string tvb<sig, real, real2>::description() const
   {
   std::ostringstream sout;
   sout << "Time-Varying Block Code (q=" << q << ", ";
   switch (codebook_type)
      {
      case codebook_sparse:
         sout << "sparse codebook n=" << get_codeword_length();
         break;

      case codebook_random:
         sout << "random codebooks n=" << get_codeword_length();
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

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   sout << ", thresholds " << th_inner << "/" << th_outer;
   sout << ", Pr=" << Pr;
   sout << ", normalized";
   sout << ", batch interface";
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
   sout << "), ";
   if (fba_ptr)
      sout << fba_ptr->description();
   else
      sout << "FBA object not initialized";
   return sout.str();
   }

// object serialization - saving

template <class sig, class real, class real2>
std::ostream& tvb<sig, real, real2>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 11 << std::endl;
   sout << "# Inner threshold" << std::endl;
   sout << th_inner << std::endl;
   sout << "# Outer threshold" << std::endl;
   sout << th_outer << std::endl;
   sout << "# Probability of channel event outside chosen limits" << std::endl;
   sout << Pr << std::endl;
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
   sout << "# q" << std::endl;
   sout << q << std::endl;
   sout << "# codebook type (0=sparse, 1=random, 2=user[seq], 3=user[ran])"
         << std::endl;
   sout << codebook_type << std::endl;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         sout << "# codeword length (n)" << std::endl;
         sout << get_codeword_length() << std::endl;
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
            sout << "#: codeword length (table " << i << ")" << std::endl;
            sout << codebook_tables(i, 0).size() << std::endl;
            sout << "#: codebook entries (table " << i << ")" << std::endl;
            for (int d = 0; d < q; d++)
               {
               codebook_tables(i, d).serialize(sout, ' ');
               //sout << std::endl;
               }
            }
         break;

      default:
         failwith("Unsupported codebook type");
         break;
      }
   sout << "# marker type (0=zero, 1=random)" << std::endl;
   sout << marker_type << std::endl;
   switch (marker_type)
      {
      case marker_zero:
      case marker_random:
         break;

      default:
         failwith("Unsupported marker sequence type");
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
 *
 * \version 7 Added option for channel-symbol-level priors
 *
 * \version 8 Removed option for channel-symbol-level priors
 *
 * \version 9 Added probability of channel event outside chosen limits
 *
 * \version 10 Removed normalization and batch flags (both always enabled)
 *
 * \version 11 Added support for codebooks with different codeword length;
 *      removed internal representation of user-defined marker sequences
 *      (use separate codebooks instead)
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
   // read probability of channel event outside chosen limits
   if (version >= 9)
      sin >> libbase::eatcomments >> Pr >> libbase::verify;
   else
      Pr = 1e-10;
   // read decoder parameters
   if (version >= 2 && version <= 9)
      {
      bool norm, batch;
      sin >> libbase::eatcomments >> norm >> libbase::verify;
      sin >> libbase::eatcomments >> batch >> libbase::verify;
      if(!norm)
         std::cerr << "WARNING: no-normalization not supported." << std::endl;
      if(!batch)
         std::cerr << "WARNING: non-batch interface not supported." << std::endl;
      }
   if (version >= 2)
      sin >> libbase::eatcomments >> flags.lazy >> libbase::verify;
   else
      flags.lazy = true;
   if (version == 7)
      {
      bool splitpriors;
      sin >> libbase::eatcomments >> splitpriors >> libbase::verify;
      assertalways(splitpriors == false);
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
   int n = 0;
   if (version < 11)
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
         {
         // sparse codebooks defined only for GF(2)
         assertalways(field_utils<sig>::elements() == 2);
         // get codeword length
         if (version >= 11)
            sin >> libbase::eatcomments >> n >> libbase::verify;
         // create sparse codebook and copy over
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
         // get codeword length
         if (version >= 11)
            sin >> libbase::eatcomments >> n >> libbase::verify;
         // Initialize space for single codebook
         libbase::allocate(codebook_tables, 1, q, n);
         // Codebook gets re-generated for each symbol index on advance()
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
            // read codeword length
            if (version >= 11)
               sin >> libbase::eatcomments >> n >> libbase::verify;
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
         // Nothing to do here. Marker gets re-generated and applied as needed
         // for each symbol index on advance()
         break;

      case marker_user_sequential:
      case marker_user_random:
         {
         // only valid if we had a single user-supplied codebook
         assertalways(
               codebook_type == codebook_user_sequential
                     || codebook_type == codebook_user_random);
         assertalways(codebook_tables.size().rows() == 1);
         assertalways(version <= 10);
         // read count of marker vectors
         sin >> libbase::eatcomments >> temp >> libbase::verify;
         // make a local copy of existing codebook
         array1vs_t codebook_s;
         libbase::allocate(codebook_s, q, n);
         for (int d = 0; d < q; d++)
            codebook_s(d) = codebook_tables(0,d);
         // duplicate to be able to apply each marker
         codebook_tables.init(temp, q);
         for (int i = 0; i < num_codebooks(); i++)
            copycodebook(i, codebook_s);
         // read marker vectors from stream and apply to corresponding codebook
         array1s_t marker_vector;
         marker_vector.init(n);
         sin >> libbase::eatcomments;
         for (int i = 0; i < temp; i++)
            {
            // read marker vector i
            marker_vector.serialize(sin);
            libbase::verify(sin);
            // apply to codebook i
            for (int d = 0; d < q; d++)
               field_utils<sig>::add_to(codebook_tables(i, d), marker_vector);
            }
         // determine order of codebooks based on order of markers
         if (marker_type == marker_user_sequential)
            codebook_type = codebook_user_sequential;
         else if (marker_type == marker_user_random)
            codebook_type = codebook_user_random;
         // reset marker type to none
         marker_type = marker_zero;
         }
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
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define ALL_SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#ifdef USE_CUDA
#define REAL_PAIRS_SEQ \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#else
#define REAL_PAIRS_SEQ \
   ((mpgnu)(mpgnu)) \
   ((logrealfast)(logrealfast)) \
   ((double)(double)) \
   ((double)(float)) \
   ((float)(float))
#endif

/* Serialization string: tvb<type,real,real2>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | [logrealfast | mpgnu (CPU only)]
 *      real2 = float | double | [logrealfast | mpgnu (CPU only)]
 */
#define INSTANTIATE3(args) \
      template class tvb<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer tvb<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockmodem", \
            "tvb<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            tvb<BOOST_PP_SEQ_ENUM(args)>::create);

#define INSTANTIATE2(r, symbol, reals) \
      INSTANTIATE3( symbol reals )

#define INSTANTIATE1(r, symbol) \
      BOOST_PP_SEQ_FOR_EACH(INSTANTIATE2, symbol, REAL_PAIRS_SEQ)

// NOTE: we *have* to use for-each product here as we cannot nest for-each
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE1, (ALL_SYMBOL_TYPE_SEQ))

} // end namespace
