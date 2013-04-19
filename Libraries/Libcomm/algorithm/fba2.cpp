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

#include "fba2.h"
#include "pacifier.h"
#include "vectorutils.h"
#include "cputimer.h"
#include <iomanip>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show allocated memory sizes
// 3 - Show input and intermediate vectors when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// *** Internal functions - computer

// common small tasks

template <class receiver_t, class sig, class real, class real2>
real fba2<receiver_t, sig, real, real2>::get_threshold(const array2r_t& metric,
      int row, int col_min, int col_max, real factor)
   {
   const bool thresholding = (factor > real(0));
   real threshold = 0;
   if (thresholding)
      {
      for (int col = col_max; col <= col_max; col++)
         if (metric[row][col] > threshold)
            threshold = metric[row][col];
      threshold *= factor;
      }
   return threshold;
   }

template <class receiver_t, class sig, class real, class real2>
real fba2<receiver_t, sig, real, real2>::get_scale(const array2r_t& metric,
      int row, int col_min, int col_max)
   {
   real scale = 0;
   for (int col = col_min; col <= col_max; col++)
      scale += metric[row][col];
   assertalways(scale > real(0));
   scale = real(1) / scale;
   return scale;
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::normalize(array2r_t& metric, int row,
      int col_min, int col_max)
   {
   // determine the scale factor to use (each block has to do this)
   const real scale = get_scale(metric, row, col_min, col_max);
   // scale all results
   for (int col = col_min; col <= col_max; col++)
      metric[row][col] *= scale;
   }

// specialized components for decode funtions

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma_global_single(
      const array1s_t& r, const array1vd_t& app, const int i) const
   {
   // compute metric with single interface
   for (int d = 0; d < q; d++)
      for (int x = -xmax; x <= xmax; x++)
         {
         // clear gamma entries
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma.global[d][i][x][deltax] = 0;
         // limit on end-state (-xmax <= x2 <= xmax):
         //   x2-x1 <= xmax-x1
         //   x2-x1 >= -xmax-x1
         const int deltaxmin = std::max(-xmax - x, dmin);
         const int deltaxmax = std::min(xmax - x, dmax);
         for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
            gamma.global[d][i][x][deltax] = compute_gamma_single(d, i, x,
                  deltax, r, app);
         }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma_global_batch(
      const array1s_t& r, const array1vd_t& app, const int i) const
   {
   // allocate space for results
   static array1r_t ptable;
   ptable.init(2 * dxmax + 1);
   // compute metric with batch interface
   for (int d = 0; d < q; d++)
      for (int x = -xmax; x <= xmax; x++)
         {
         compute_gamma_batch(d, i, x, ptable, r, app);
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma.global[d][i][x][deltax] = ptable(dxmax + deltax);
         }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma_local_single(
      const array1s_t& r, const array1vd_t& app, const int i) const
   {
   // compute metric with single interface
   for (int d = 0; d < q; d++)
      for (int x = -xmax; x <= xmax; x++)
         {
         // clear gamma entries
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma.local[d][x][deltax] = 0;
         // limit on end-state (-xmax <= x2 <= xmax):
         //   x2-x1 <= xmax-x1
         //   x2-x1 >= -xmax-x1
         const int deltaxmin = std::max(-xmax - x, dmin);
         const int deltaxmax = std::min(xmax - x, dmax);
         for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
            gamma.local[d][x][deltax] = compute_gamma_single(d, i, x, deltax, r,
                  app);
         }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma_local_batch(
      const array1s_t& r, const array1vd_t& app, const int i) const
   {
   // allocate space for results
   static array1r_t ptable;
   ptable.init(2 * dxmax + 1);
   // compute metric with batch interface
   for (int d = 0; d < q; d++)
      for (int x = -xmax; x <= xmax; x++)
         {
         compute_gamma_batch(d, i, x, ptable, r, app);
         for (int deltax = dmin; deltax <= dmax; deltax++)
            gamma.local[d][x][deltax] = ptable(dxmax + deltax);
         }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_message_app(array1vr_t& ptable,
      const int i) const
   {
   // local flag for path thresholding
   const bool thresholding = (th_outer > real(0));
   // determine the strongest path at this point
   const real threshold = get_threshold(alpha, i, -xmax, xmax, th_outer);
   for (int d = 0; d < q; d++)
      {
      // initialize result holder
      real p = 0;
      for (int x1 = -xmax; x1 <= xmax; x1++)
         {
         // cache this alpha value in a register
         const real this_alpha = alpha[i][x1];
         // ignore paths below a certain threshold
         if (thresholding && this_alpha < threshold)
            continue;
         // limits on deltax can be combined as (c.f. allocate() for details):
         //   x2-x1 <= dmax
         //   x2-x1 >= dmin
         const int x2min = std::max(-xmax, dmin + x1);
         const int x2max = std::min(xmax, dmax + x1);
         for (int x2 = x2min; x2 <= x2max; x2++)
            {
            real temp = this_alpha;
            temp *= beta[i + 1][x2];
            temp *= get_gamma(d, i, x1, x2 - x1);
            p += temp;
            }
         }
      // store result
      ptable(i)(d) = p;
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_message_app(
      array1vr_t& ptable) const
   {
   libbase::pacifier progress("FBA Results");
   // Initialise result vector (one symbol per timestep)
   libbase::allocate(ptable, N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // pre-compute local gamma values, if necessary
      if (!flags.lazy && !flags.globalstore)
         work_gamma_local(r, app, i);
      // compute partial result
      work_message_app(ptable, i);
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
#ifndef NDEBUG
   // show cache statistics
   std::cerr << "FBA Cache Usage: "
         << 100 * gamma_misses / double(cached.global.num_elements()) << "%"
         << std::endl;
   std::cerr << "FBA Cache Reuse: "
         << gamma_calls / double(gamma_misses * q * (dmax - dmin + 1)) << "x"
         << std::endl;
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_state_app(array1r_t& ptable,
      const int i) const
   {
   assert(i >= 0 && i <= N);
   // compute posterior probabilities for given index
   ptable.init(2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      ptable(xmax + x) = alpha[i][x] * beta[i][x];
   }

// *** Internal functions - main

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::allocate()
   {
   // flag the state of the arrays
   initialised = true;

   // determine allowed limits on deltax:
   // limits on insertions and deletions:
   //   x2-x1 <= n*I
   //   x2-x1 >= -n
   // limits on introduced drift in this section:
   // (necessary for forward recursion on extracted segment)
   //   x2-x1 <= dxmax
   //   x2-x1 >= -dxmax
   // the above two sets of limits can be combined as:
   //   x2-x1 <= min(n*I, dxmax) = dmax
   //   x2-x1 >= max(-n, -dxmax) = dmin
   dmin = std::max(-n, -dxmax);
   dmax = std::min(n * I, dxmax);
   // alpha needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   alpha.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);
   beta.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);

   if (flags.globalstore)
      {
      // gamma needs indices (d,i,x,deltax) where d in [0, q-1], i in [0, N-1]
      // x in [-xmax, xmax], and deltax in [dmin, dmax]
      gamma.global.resize(
            boost::extents[q][N][range(-xmax, xmax + 1)][range(dmin, dmax + 1)]);
      gamma.local.resize(boost::extents[0][0][0]);
      }
   else
      {
      // gamma needs indices (d,x,deltax) where d in [0, q-1]
      // x in [-xmax, xmax], and deltax in [dmin, dmax]
      gamma.local.resize(
            boost::extents[q][range(-xmax, xmax + 1)][range(dmin, dmax + 1)]);
      gamma.global.resize(boost::extents[0][0][0][0]);
      }
   // need to keep track only if we're caching lazy computations
   if (flags.lazy)
      {
      if (flags.globalstore)
         {
         // cached needs indices (i,x) where i in [0, N-1] and x in [-xmax, xmax]
         cached.global.resize(boost::extents[N][range(-xmax, xmax + 1)]);
         cached.local.resize(boost::extents[0]);
         }
      else
         {
         // cached needs indices (x) where x in [-xmax, xmax]
         cached.local.resize(boost::extents[range(-xmax, xmax + 1)]);
         cached.global.resize(boost::extents[0][0]);
         }
      }
   else
      {
      cached.global.resize(boost::extents[0][0]);
      cached.local.resize(boost::extents[0]);
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
   const size_t bytes_used =
         sizeof(bool)
               * (cached.global.num_elements() + cached.local.num_elements())
               + sizeof(real)
                     * (alpha.num_elements() + beta.num_elements()
                           + gamma.global.num_elements()
                           + gamma.local.num_elements());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(old_precision);
   std::cerr.flags(old_flags);
#endif

#ifndef NDEBUG
   // determine required space for inner metric table (Jiao-Armand method)
   size_t entries = 0;
   for (int delta = dmin; delta <= dmax; delta++)
      entries += (1 << (delta + n));
   std::cerr << "Jiao-Armand Table Size: "
         << q * entries * sizeof(float) / double(1 << 20) << "MiB" << std::endl;
#endif

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "dmax = " << dmax << std::endl;
   std::cerr << "dmin = " << dmin << std::endl;
   std::cerr << "alpha = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << alpha.num_elements() << std::endl;
   std::cerr << "beta = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << beta.num_elements() << std::endl;
   if (flags.globalstore)
      {
      std::cerr << "gamma = " << q << "x" << N << "x" << 2 * xmax + 1 << "x"
      << dmax - dmin + 1 << " = " << gamma.global.num_elements()
      << std::endl;
      if (flags.lazy)
         {
         std::cerr << "cached = " << N << "x" << 2 * xmax + 1 << " = "
         << cached.global.num_elements() << std::endl;
         }
      }
   else
      {
      std::cerr << "gamma = " << q << "x" << 2 * xmax + 1 << "x" << dmax - dmin
      + 1 << " = " << gamma.local.num_elements() << std::endl;
      if (flags.lazy)
         {
         std::cerr << "cached = " << 2 * xmax + 1 << " = "
         << cached.local.num_elements() << std::endl;
         }
      }
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::free()
   {
   alpha.resize(boost::extents[0][0]);
   beta.resize(boost::extents[0][0]);
   gamma.global.resize(boost::extents[0][0][0][0]);
   gamma.local.resize(boost::extents[0][0][0]);
   cached.global.resize(boost::extents[0][0]);
   cached.local.resize(boost::extents[0]);
   // flag the state of the arrays
   initialised = false;
   }

// helper methods

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::reset_cache() const
   {
   // initialise array and cache flags
   if (flags.globalstore)
      {
      gamma.global = real(0);
      cached.global = false;
      }
   else
      {
      gamma.local = real(0);
      cached.local = false;
      }
#ifndef NDEBUG
   // reset cache counters
   gamma_calls = 0;
   gamma_misses = 0;
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::print_gamma(std::ostream& sout) const
   {
   // gamma has indices (d,i,x,deltax) where:
   //    d in [0, q-1], i in [0, N-1], x in [-xmax, xmax], and
   //    deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
   for (int i = 0; i < N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int d = 0; d < q; d++)
         {
         sout << "d = " << d << ":" << std::endl;
         for (int x = -xmax; x <= xmax; x++)
            {
            for (int deltax = dmin; deltax <= dmax; deltax++)
               sout << '\t' << gamma.global[d][i][x][deltax];
            sout << std::endl;
            }
         }
      }
   }

// decode functions

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma(const array1s_t& r,
      const array1vd_t& app)
   {
   assert(initialised);
#ifndef NDEBUG
   if (app.size() == 0)
      std::cerr << "DEBUG (fba2): Empty APP table." << std::endl;
#endif
   if (flags.lazy)
      {
      // keep a copy of received vector and a-priori statistics
      This::r = r;
      This::app = app;
      // reset cache values
      reset_cache();
      }
   else if (flags.globalstore)
      {
      // global pre-computation of gamma values
      libbase::pacifier progress("FBA Gamma");
      // compute metric with independent interface
      for (int i = 0; i < N; i++)
         {
         std::cerr << progress.update(i, N);
         // compute partial result
         work_gamma_global(r, app, i);
         }
      std::cerr << progress.update(N, N);
#if DEBUG>=3
      std::cerr << "gamma = " << std::endl;
      print_gamma(std::cerr);
#endif
      }
   else
      {
      // local pre-computation is integrated with alpha/beta/results computation
      // keep a copy of received vector and a-priori statistics
      This::r = r;
      This::app = app;
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_alpha(const array1d_t& sof_prior,
      const int i)
   {
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // determine the strongest path at this point
   const real threshold = get_threshold(alpha, i - 1, -xmax, xmax, th_inner);
   for (int x1 = -xmax; x1 <= xmax; x1++)
      {
      // cache previous alpha value in a register
      const real prev_alpha = alpha[i - 1][x1];
      // ignore paths below a certain threshold
      if (thresholding && prev_alpha < threshold)
         continue;
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= dmax
      //   x2-x1 >= dmin
      const int x2min = std::max(-xmax, dmin + x1);
      const int x2max = std::min(xmax, dmax + x1);
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         // NOTE: we're repeating the loop on x2, so we need to increment this
         real this_alpha = alpha[i][x2];
         for (int d = 0; d < q; d++)
            {
            real temp = prev_alpha;
            temp *= get_gamma(d, i - 1, x1, x2 - x1);
            this_alpha += temp;
            }
         alpha[i][x2] = this_alpha;
         }
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_alpha(const array1d_t& sof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Alpha");
   // initialise array:
   alpha = real(0);
   // set initial drift distribution
   for (int x = -xmax; x <= xmax; x++)
      alpha[0][x] = real(sof_prior(xmax + x));
   // normalize if requested
   if (flags.norm)
      normalize_alpha(0);
   // compute remaining matrix values
   for (int i = 1; i <= N; i++)
      {
      std::cerr << progress.update(i - 1, N);
      // pre-compute local gamma values, if necessary
      if (!flags.lazy && !flags.globalstore)
         work_gamma_local(r, app, i - 1);
      // compute partial result
      work_alpha(sof_prior, i);
      // normalize if requested
      if (flags.norm)
         normalize_alpha(i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   std::cerr << "alpha = " << alpha << std::endl;
   // show after alpha if computing lazily
   if (flags.globalstore && flags.lazy)
      {
      std::cerr << "gamma = " << std::endl;
      print_gamma(std::cerr);
      }
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_beta(const array1d_t& eof_prior,
      const int i)
   {
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // determine the strongest path at this point
   const real threshold = get_threshold(beta, i + 1, -xmax, xmax, th_inner);
   for (int x1 = -xmax; x1 <= xmax; x1++)
      {
      real this_beta = 0;
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= dmax
      //   x2-x1 >= dmin
      const int x2min = std::max(-xmax, dmin + x1);
      const int x2max = std::min(xmax, dmax + x1);
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         // cache next beta value in a register
         const real next_beta = beta[i + 1][x2];
         // ignore paths below a certain threshold
         if (thresholding && next_beta < threshold)
            continue;
         for (int d = 0; d < q; d++)
            {
            real temp = next_beta;
            temp *= get_gamma(d, i, x1, x2 - x1);
            this_beta += temp;
            }
         }
      beta[i][x1] = this_beta;
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_beta(const array1d_t& eof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Beta");
   // initialise array:
   // NOTE: technically we should not need to do this, as we're initializing
   //       this_beta for every value
   beta = real(0);
   // set final drift distribution
   for (int x = -xmax; x <= xmax; x++)
      beta[N][x] = real(eof_prior(xmax + x));
   // normalize if requested
   if (flags.norm)
      normalize_beta(N);
   // compute remaining matrix values
   for (int i = N - 1; i >= 0; i--)
      {
      std::cerr << progress.update(N - 1 - i, N);
      // pre-compute local gamma values, if necessary
      if (!flags.lazy && !flags.globalstore)
         work_gamma_local(r, app, i);
      // compute partial result
      work_beta(eof_prior, i);
      // normalize if requested
      if (flags.norm)
         normalize_beta(i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   std::cerr << "beta = " << beta << std::endl;
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_results(array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post) const
   {
   assert(initialised);
   // compute APPs of message
   work_message_app(ptable);
   // compute APPs of sof/eof state values
   work_state_app(sof_post, 0);
   work_state_app(eof_post, N);
#if DEBUG>=3
   // show output data
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
#endif
   }

// User procedures

// Initialization

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::init(int N, int n, int q, int I,
      int xmax, int dxmax, double th_inner, double th_outer, bool norm,
      bool batch, bool lazy, bool globalstore)
   {
   // if any parameters that effect memory have changed, release memory
   if (initialised
         && (N != This::N || n != This::n || q != This::q || I != This::I
               || xmax != This::xmax || dxmax != This::dxmax
               || lazy != This::flags.lazy
               || globalstore != This::flags.globalstore))
      free();
   // code parameters
   assert(N > 0);
   assert(n > 0);
   This::N = N;
   This::n = n;
   assert(q > 1);
   This::q = q;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   assert(dxmax > 0);
   This::I = I;
   This::xmax = xmax;
   This::dxmax = dxmax;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   This::th_inner = real(th_inner);
   This::th_outer = real(th_outer);
   // decoding mode parameters
   This::flags.norm = norm;
   This::flags.batch = batch;
   This::flags.lazy = lazy;
   This::flags.globalstore = globalstore;
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
 *       xmax and padded to a total length of tau + 2*xmax, where tau is the
 *       length of the transmitted frame. This avoids special handling for such
 *       vectors.
 *
 * \note Offset is the same as for stream_modulator.
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::decode(
      libcomm::instrumented& collector, const array1s_t& r,
      const array1d_t& sof_prior, const array1d_t& eof_prior,
      const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
      array1r_t& eof_post, const int offset)
   {
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << N << std::endl;
   std::cerr << "n = " << n << std::endl;
   std::cerr << "q = " << q << std::endl;
   std::cerr << "I = " << I << std::endl;
   std::cerr << "xmax = " << xmax << std::endl;
   std::cerr << "dxmax = " << dxmax << std::endl;
   std::cerr << "th_inner = " << th_inner << std::endl;
   std::cerr << "th_outer = " << th_outer << std::endl;
   std::cerr << "norm = " << flags.norm << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   // Initialise memory if necessary
   if (!initialised)
      allocate();
   // Validate sizes and offset
   const int tau = N * n;
   assertalways(offset == xmax);
   assertalways(r.size() == tau + 2 * xmax);
   assertalways(sof_prior.size() == 2 * xmax + 1);
   assertalways(eof_prior.size() == 2 * xmax + 1);
#if DEBUG>=3
   // show input data
   std::cerr << "r = " << r << std::endl;
   std::cerr << "app = " << app << std::endl;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
#endif

   // Gamma
   libbase::cputimer tg("t_gamma");
   work_gamma(r, app);
   collector.add_timer(tg);
   // Alpha
   libbase::cputimer ta("t_alpha");
   work_alpha(sof_prior);
   collector.add_timer(ta);
   // Beta
   libbase::cputimer tb("t_beta");
   work_beta(eof_prior);
   collector.add_timer(tb);
   // Compute results
   libbase::cputimer tr("t_results");
   work_results(ptable, sof_post, eof_post);
   collector.add_timer(tr);

   // Add values for limits that depend on channel conditions
   collector.add_timer(I, "c_I");
   collector.add_timer(xmax, "c_xmax");
   collector.add_timer(dxmax, "c_dxmax");
   // Add memory usage
   collector.add_timer(sizeof(real) * alpha.num_elements(), "m_alpha");
   collector.add_timer(sizeof(real) * beta.num_elements(), "m_beta");
   collector.add_timer(sizeof(real) * gamma.global.num_elements(), "m_gamma");
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
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::get_drift_pdf(
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

#include "gf.h"
#include "mpgnu.h"
#include "logrealfast.h"
#include "modem/tvb-receiver.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)(mpgnu)(logrealfast)

// *** Instantiations for tvb: bool and gf types only ***

#define INSTANTIATE(r, args) \
      template class fba2<tvb_receiver<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)> ;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
