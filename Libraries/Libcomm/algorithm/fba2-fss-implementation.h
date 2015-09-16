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

#include "fba2-fss.h"
#include "pacifier.h"
#include "vectorutils.h"
#include "cputimer.h"
#include <iomanip>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show allocated memory sizes
// 3 - Show input and intermediate vectors when decoding
// 4 - Show alpha/beta matrices before normalization
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// *** Internal functions - computer

// common small tasks

template <class receiver_t, class sig, class real, class real2, bool globalstore>
real fba2_fss<receiver_t, sig, real, real2, globalstore>::get_scale(
      const array3r_t& metric, int i, int m_min, int m_max)
   {
   real scale = 0;
   for (int m = m_min; m <= m_max; m++)
      for (int delta = 0; delta <= 1; delta++)
         scale += metric[i][m][delta];
   assertalways(scale > real(0));
   scale = real(1) / scale;
   return scale;
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::normalize(
      array3r_t& metric, int i, int m_min, int m_max)
   {
   // determine the scale factor to use (each block has to do this)
   const real scale = get_scale(metric, i, m_min, m_max);
   // scale all results
   for (int m = m_min; m <= m_max; m++)
      for (int delta = 0; delta <= 1; delta++)
         metric[i][m][delta] *= scale;
   }

// decode functions - partial computations

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_alpha(
      const int i)
   {
   for (int m1 = Zmin; m1 <= Zmax; m1++)
      for (int delta1 = 0; delta1 <= 1; delta1++)
         {
         // cache previous alpha value in a register
         const real prev_alpha = alpha[i - 1][m1][delta1];
         // Determine limits over this sequence
         // a) upper limit is the same as global limit
         // b) lower limit is bounded by ⌈n/2⌉ (one deletion per output bit)
         const int m2max = Zmax;
         const int m2min = std::max(Zmin, m1 - int(std::ceil(cw_length(i-1) / 2.0)));
         for (int m2 = m2min; m2 <= m2max; m2++)
            for (int delta2 = 0; delta2 <= 1; delta2++)
               {
               // NOTE: we're repeating the loop on m2,delta2 so we need to increment this
               real this_alpha = alpha[i][m2][delta2];
               for (int d = 0; d < q; d++)
                  {
                  real temp = prev_alpha;
                  temp *= gamma_storage_entry(d, i - 1, m1, delta1, m2, delta2);
                  this_alpha += temp;
                  }
               alpha[i][m2][delta2] = this_alpha;
               }
         }
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_beta(const int i)
   {
   for (int m1 = Zmin; m1 <= Zmax; m1++)
      for (int delta1 = 0; delta1 <= 1; delta1++)
         {
         real this_beta = 0;
         // Determine limits over this sequence
         // a) upper limit is the same as global limit
         // b) lower limit is bounded by ⌈n/2⌉ (one deletion per output bit)
         const int m2max = Zmax;
         const int m2min = std::max(Zmin, m1 - int(std::ceil(cw_length(i) / 2.0)));
         for (int m2 = m2min; m2 <= m2max; m2++)
            for (int delta2 = 0; delta2 <= 1; delta2++)
               {
               // cache next beta value in a register
               const real next_beta = beta[i + 1][m2][delta2];
               for (int d = 0; d < q; d++)
                  {
                  real temp = next_beta;
                  temp *= gamma_storage_entry(d, i, m1, delta1, m2, delta2);
                  this_beta += temp;
                  }
               }
         beta[i][m1][delta1] = this_beta;
         }
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_message_app(
      array1vr_t& ptable, const int i) const
   {
   for (int d = 0; d < q; d++)
      {
      // initialize result holder
      real p = 0;
      for (int m1 = Zmin; m1 <= Zmax; m1++)
         for (int delta1 = 0; delta1 <= 1; delta1++)
            {
            // cache this alpha value in a register
            const real this_alpha = alpha[i][m1][delta1];
            // Determine limits over this sequence
            // a) upper limit is the same as global limit
            // b) lower limit is bounded by ⌈n/2⌉ (one deletion per output bit)
            const int m2max = Zmax;
            const int m2min = std::max(Zmin, m1 - int(std::ceil(cw_length(i) / 2.0)));
            for (int m2 = m2min; m2 <= m2max; m2++)
               for (int delta2 = 0; delta2 <= 1; delta2++)
                  {
                  real temp = this_alpha;
                  temp *= beta[i + 1][m2][delta2];
                  temp *= gamma_storage_entry(d, i, m1, delta1, m2, delta2);
                  p += temp;
                  }
            }
      // store result
      ptable(i)(d) = p;
      }
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_state_app(array1r_t& ptable,
      const int i) const
   {
   assert(i >= 0 && i <= N);
   // compute posterior probabilities for given index
   ptable.init(Zmax - Zmin + 1);
   for (int x = Zmin; x <= Zmax; x++)
      {
      real r = 0;
      for (int delta = 0; delta <= 1; delta++)
         r += alpha[i][x][delta] * beta[i][x][delta];
      ptable(x - Zmin) = r;
      }
   }

// *** Internal functions - main

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::allocate()
   {
   // flag the state of the arrays
   initialised = true;

   /* alpha and beta need indices (i,m,delta) where
    * i in [0, N]
    * m in [Zmin, Zmax]
    * delta in [0, 1]
    */
   typedef boost::multi_array_types::extent_range range;
   alpha.resize(boost::extents[N + 1][range(Zmin, Zmax + 1)][2]);
   beta.resize(boost::extents[N + 1][range(Zmin, Zmax + 1)][2]);

   if (globalstore)
      {
      /* gamma needs indices (i,m1,delta1,d,m2,delta2) where
       * i in [0, N-1]
       * m1,m2 in [Zmin, Zmax]
       * delta1,delta2 in [0, 1]
       * d in [0, q-1]
       */
      gamma.global.resize(
            boost::extents[N][range(Zmin, Zmax + 1)][2][q][range(Zmin, Zmax + 1)][2]);
      gamma.local.resize(boost::extents[0][0][0][0][0]);
      }
   else
      {
      /* gamma needs indices (m1,delta1,d,m2,delta2) where
       * m1,m2 in [Zmin, Zmax]
       * delta1,delta2 in [0, 1]
       * d in [0, q-1]
       */
      gamma.local.resize(
            boost::extents[range(Zmin, Zmax + 1)][2][q][range(Zmin, Zmax + 1)][2]);
      gamma.global.resize(boost::extents[0][0][0][0][0][0]);
      }

   // if this is not the first time, skip the rest
   static bool first_time = true;
   if (!first_time)
      return;
   first_time = false;

#ifndef NDEBUG
   // set required format, storing previous settings
   const std::ios::fmtflags old_flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const std::streamsize old_precision = std::cerr.precision(1);
   // determine memory occupied and tell user
   size_t bytes_used = 0;
   bytes_used += sizeof(real) * alpha.num_elements();
   bytes_used += sizeof(real) * beta.num_elements();
   bytes_used += sizeof(real) * gamma.global.num_elements();
   bytes_used += sizeof(real) * gamma.local.num_elements();
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(old_precision);
   std::cerr.flags(old_flags);
#endif

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "alpha = " << N + 1 << "×" << Zmax - Zmin + 1 << "×2 = "
   << alpha.num_elements() << std::endl;
   std::cerr << "beta = " << N + 1 << "×" << Zmax - Zmin + 1 << "×2 = "
   << beta.num_elements() << std::endl;
   if (globalstore)
      {
      std::cerr << "gamma = " << q << "×" << N << "×" << Zmax - Zmin + 1 << "×2×"
      << Zmax - Zmin + 1 << "×2 = " << gamma.global.num_elements()
      << std::endl;
      }
   else
      {
      std::cerr << "gamma = " << q << "×" << Zmax - Zmin + 1 << "×2×" << Zmax - Zmin
      + 1 << "×2 = " << gamma.local.num_elements() << std::endl;
      }
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::free()
   {
   alpha.resize(boost::extents[0][0][0]);
   beta.resize(boost::extents[0][0][0]);
   gamma.global.resize(boost::extents[0][0][0][0][0][0]);
   gamma.local.resize(boost::extents[0][0][0][0][0]);
   // flag the state of the arrays
   initialised = false;
   }

// helper methods

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::print_gamma(std::ostream& sout) const
   {
   sout << "gamma = " << std::endl;
   for (int i = 0; i < N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int d = 0; d < q; d++)
         {
         sout << "d = " << d << ":" << std::endl;
         for (int m1 = Zmin; m1 <= Zmax; m1++)
            for (int delta1 = 0; delta1 <= 1; delta1++)
               {
               sout << "m1 = " << m1 << ", delta1 = " << delta1 << ":"
                     << std::endl;
               for (int delta2 = 0; delta2 <= 1; delta2++)
                  {
                  for (int m2 = Zmin; m2 <= Zmax; m2++)
                     sout << '\t' << gamma.global[i][m1][delta1][d][m2][delta2];
                  sout << std::endl;
                  }
               }
         }
      }
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::print_metric(
      std::ostream& sout, const array3r_t& metric) const
   {
   for (int i = 0; i <= N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int delta = 0; delta <= 1; delta++)
         {
         for (int m = Zmin; m <= Zmax; m++)
            sout << '\t' << metric[i][m][delta];
         sout << std::endl;
         }
      }
   }

// decode functions - global path

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_gamma(const array1s_t& r,
      const array1vd_t& app)
   {
   assert(initialised);
#ifndef NDEBUG
   if (app.size() == 0)
      std::cerr << "DEBUG (fba2-fss): Empty APP table." << std::endl;
#endif
   // global pre-computation of gamma values
   libbase::pacifier progress("FBA Gamma");
   // compute metric at each symbol index
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // compute partial result
      work_gamma(r, app, i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   print_gamma(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_alpha_and_beta(
      const array1d_t& sof_prior, const array1d_t& eof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Alpha + Beta");
   // initialise arrays:
   // NOTE: technically unnecessary for beta, as we initialize this_beta for every value
   alpha = real(0);
   beta = real(0);
   // set initial and final drift distribution
   for (int x = Zmin; x <= Zmax; x++)
      {
      alpha[0][x][0] = real(sof_prior(x - Zmin));
      beta[N][x][0] = real(eof_prior(x - Zmin));
      }
   // normalize
   normalize_alpha(0);
   normalize_beta(N);
   // compute remaining matrix values
   for (int i = 1; i <= N; i++)
      {
      std::cerr << progress.update(i - 1, N);
      // compute partial result
      work_alpha(i);
      work_beta(N - i);
#if DEBUG>=4
      std::cerr << "pre-norm metrics:" << std::endl;
      print_alpha(std::cerr);
      print_beta(std::cerr);
#endif
      // normalize
      normalize_alpha(i);
      normalize_beta(N - i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   print_alpha(std::cerr);
   print_beta(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_results(array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post) const
   {
   assert(initialised);
   libbase::pacifier progress("FBA Results");
   // Initialise result vector:
   // ptable(i,d) = posterior prob. of having transmitted symbol 'd' at time 'i'
   libbase::allocate(ptable, N, q);
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // compute partial result
      work_message_app(ptable, i);
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
   // compute APPs of sof/eof state values
   work_state_app(sof_post, 0);
   work_state_app(eof_post, N);
#if DEBUG>=3
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
#endif
   }

// decode functions - local path

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_alpha(const array1d_t& sof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Alpha");
   // initialise array:
   alpha = real(0);
   // set initial drift distribution
   for (int x = Zmin; x <= Zmax; x++)
      alpha[0][x][0] = real(sof_prior(x - Zmin));
   // normalize
   normalize_alpha(0);
   // compute remaining matrix values
   for (int i = 1; i <= N; i++)
      {
      std::cerr << progress.update(i - 1, N);
      // local storage
      if (!globalstore)
         {
         // pre-compute local gamma values
         work_gamma(r, app, i - 1);
         }
      // compute partial result
      work_alpha(i);
      // normalize
      normalize_alpha(i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   print_alpha(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::work_beta_and_results(
      const array1d_t& eof_prior, array1vr_t& ptable, array1r_t& sof_post,
      array1r_t& eof_post)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Beta + Results");
   // Initialise result vector:
   // ptable(i,d) = posterior prob. of having transmitted symbol 'd' at time 'i'
   libbase::allocate(ptable, N, q);
   // initialise array:
   // NOTE: technically unnecessary for beta, as we initialize this_beta for every value
   beta = real(0);
   // set final drift distribution
   for (int x = Zmin; x <= Zmax; x++)
      beta[N][x][0] = real(eof_prior(x - Zmin));
   // normalize
   normalize_beta(N);
   // compute remaining matrix values
   for (int i = N - 1; i >= 0; i--)
      {
      std::cerr << progress.update(N - 1 - i, N);
      // local storage
      if (!globalstore)
         {
         // pre-compute local gamma values
         work_gamma(r, app, i);
         }
      // compute partial result
      work_beta(i);
      // normalize
      normalize_beta(i);
      // compute partial result
      work_message_app(ptable, i);
      }
   std::cerr << progress.update(N, N);
   // compute APPs of sof/eof state values
   work_state_app(sof_post, 0);
   work_state_app(eof_post, N);
#if DEBUG>=3
   print_beta(std::cerr);
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
#endif
   }

// User procedures

// Initialization

template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::init(int N, int q,
      int mtau_min, int mtau_max, int mn_min, int mn_max, int m1_min,
      int m1_max, double th_inner, double th_outer,
      const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer)
   {
   // Initialize our embedded metric computer with unchanging elements
   // (needs to happen before fba initialization)
   this->receiver.init(computer);
   // if any parameters that effect memory have changed, release memory
   if (initialised
         && (N != this->N || q != this->q
               || mtau_min != this->Zmin || mtau_max != this->Zmax))
      free();
   // code parameters
   assert(N > 0);
   this->N = N;
   assert(q > 1);
   this->q = q;
   // decoder parameters
   assert(mtau_min <= 0);
   assert(mtau_max >= 0);
   this->Zmin = mtau_min;
   this->Zmax = mtau_max;
   // path truncation parameters
   assert(th_inner == 0 && th_outer == 0);
   }

/*!
 * \brief Frame decode cycle
 * \param[in] collector Reference to (instrumented) results collector object
 * \param[in] r Received frame
 * \param[in] sof_prior Prior probabilities for start-of-frame position
 *                      (zero-index matches zero-index of r)
 * \param[in] eof_prior Prior probabilities for end-of-frame position
 *                      (zero-index matches tau-index of r, where tau is the
 *                      length of the transmitted frame)
 * \param[in] app A-Priori Probabilities for message
 * \param[out] ptable Posterior Probabilities for message
 * \param[out] sof_post Posterior probabilities for start-of-frame position
 *                      (indexing same as prior)
 * \param[out] eof_post Posterior probabilities for end-of-frame position
 *                      (indexing same as prior)
 * \param[in] offset Index offset for prior, post, and r vectors
 *
 * \note If APP table is empty, it is assumed that symbols are equiprobable.
 *
 * \note Priors for start and end-of-frame *must* be supplied; in the case of a
 *       received frame with exactly known boundaries, this must be offset by
 *       Zmax and padded to a total length of tau + Zmax-Zmin, where tau is the
 *       length of the transmitted frame. This avoids special handling for such
 *       vectors.
 *
 * \note Offset is the same as for stream_modulator.
 */
template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::decode(
      libcomm::instrumented& collector, const array1s_t& r,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
      array1r_t& eof_post, const int offset)
   {
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << N << std::endl;
   std::cerr << "q = " << q << std::endl;
   std::cerr << "Zmin = " << Zmin << std::endl;
   std::cerr << "Zmax = " << Zmax << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
   // show input data
   std::cerr << "r = " << r << std::endl;
   std::cerr << "app = " << app << std::endl;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
#endif
   // Initialise memory if necessary
   if (!initialised)
      allocate();
   // Validate sizes and offset
   assertalways(offset == -Zmin);
   assertalways(r.size() == tau + Zmax - Zmin);
   assertalways(sof_prior.size() == Zmax - Zmin + 1);
   assertalways(eof_prior.size() == Zmax - Zmin + 1);

   // Alpha + Beta + Results
   if (globalstore)
      {
      // Gamma
      libbase::cputimer tg("t_gamma");
      work_gamma(r, app);
      collector.add_timer(tg);
      // Alpha + Beta
      libbase::cputimer tab("t_alpha+beta");
      work_alpha_and_beta(sof_prior, eof_prior);
      collector.add_timer(tab);
      // Compute results
      libbase::cputimer tr("t_results");
      work_results(ptable, sof_post, eof_post);
      collector.add_timer(tr);
      }
   else
      {
      // keep a copy of received vector and a-priori statistics
      // (we need them later when computing gamma locally)
      this->r = r;
      this->app = app;
      // Alpha
      libbase::cputimer ta("t_alpha");
      work_alpha(sof_prior);
      collector.add_timer(ta);
      // Beta
      libbase::cputimer tbr("t_beta+results");
      work_beta_and_results(eof_prior, ptable, sof_post, eof_post);
      collector.add_timer(tbr);
      }

   // Add values for limits that depend on channel conditions
   collector.add_timer(Zmin, "c_Zmin");
   collector.add_timer(Zmax, "c_Zmax");
   // Add memory usage
   collector.add_timer(sizeof(real) * alpha.num_elements(), "m_alpha");
   collector.add_timer(sizeof(real) * beta.num_elements(), "m_beta");
   collector.add_timer(sizeof(real) * (gamma.global.num_elements() + gamma.local.num_elements()), "m_gamma");
   }

/*!
 * \brief Get the posterior channel drift pdf at codeword boundaries
 * \param[out] pdftable Posterior Probabilities for codeword boundaries
 *
 * Codeword boundaries are taken to include frame boundaries, such that
 * pdftable(i) corresponds to the boundary between codewords 'i' and 'i+1'.
 * This method must be called after a call to decode(), so that it can return
 * posteriors for the last transmitted frame.
 */
template <class receiver_t, class sig, class real, class real2, bool globalstore>
void fba2_fss<receiver_t, sig, real, real2, globalstore>::get_drift_pdf(
      array1vr_t& pdftable) const
   {
   assert(initialised);
   // allocate space for results
   pdftable.init(N + 1);
   // consider each time index in the order given
   for (int i = 0; i <= N; i++)
      work_state_app(pdftable(i), i);
   }

} // end namespace

/* \note There are no explicit realizations here, as for this module we need to
 * split the realizations over separate units, or g++ will run out of memory.
 * All realizations are in the fba2-fss-instXX.cpp files.
 */
