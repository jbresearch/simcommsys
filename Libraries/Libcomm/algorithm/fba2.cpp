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

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class real, class sig, bool norm>
void fba2<real, sig, norm>::allocate()
   {
   // determine limits
   dmin = std::max(-n, -dxmax);
   dmax = std::min(n * I, dxmax);
   // alpha needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   alpha.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);
   beta.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);
   // dynamically decide whether we want to use the gamma cache or not
   // decision is hardwired: use if memory requirement < 750MB
   const libbase::int64s bytes_required = sizeof(real) * (q * N
         * (2 * xmax + 1) * (dmax - dmin + 1));
   cache_enabled = bytes_required < (750 << 20);
   // gamma needs indices (d,i,x,deltax) where d in [0, q-1], i in [0, N-1]
   // x in [-xmax, xmax], and deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
   if (cache_enabled)
      {
      gamma.resize(boost::extents[q][N][range(-xmax, xmax + 1)][range(dmin,
            dmax + 1)]);
      cached.resize(boost::extents[N][range(-xmax, xmax + 1)][range(dmin, dmax
            + 1)]);
      }
   else
      {
      gamma.resize(boost::extents[0][0][0][0]);
      cached.resize(boost::extents[0][0][0]);
      std::cerr << "FBA Cache Disabled, Required: " << bytes_required
            / double(1 << 20) << "MiB" << std::endl;
      }
   // flag the state of the arrays
   initialised = true;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const int prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(bool) * cached.num_elements()
         + sizeof(real) * (alpha.num_elements() + beta.num_elements()
               + gamma.num_elements());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.setf(flags);

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "dmax = " << dmax << std::endl;
   std::cerr << "dmin = " << dmin << std::endl;
   std::cerr << "alpha = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << alpha.num_elements() << std::endl;
   std::cerr << "beta = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << beta.num_elements() << std::endl;
   std::cerr << "gamma = " << q << "x" << N << "x" << 2 * xmax + 1 << "x"
   << dmax - dmin + 1 << " = " << gamma.num_elements() << std::endl;
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class real, class sig, bool norm>
void fba2<real, sig, norm>::free()
   {
   alpha.resize(boost::extents[0][0]);
   beta.resize(boost::extents[0][0]);
   cache_enabled = false;
   gamma.resize(boost::extents[0][0][0][0]);
   cached.resize(boost::extents[0][0][0]);
   // flag the state of the arrays
   initialised = false;
   }

// Initialization

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::init(int N, int n, int q, int I, int xmax,
      int dxmax, double th_inner, double th_outer)
   {
   // if any parameters that effect memory have changed, release memory
   if (initialised && (N != This::N || n != This::n || q != This::q || I
         != This::I || xmax != This::xmax || dxmax != This::dxmax))
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
   }

// Internal procedures

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::reset_cache() const
   {
   // initialise array
   gamma = real(0);
   // initialize cache
   cached = false;
#ifndef NDEBUG
   // reset cache counters
   gamma_calls = 0;
   gamma_misses = 0;
#endif
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_gamma(const array1s_t& r,
      const array1vd_t& app)
   {
   assert(initialised);
   if (cache_enabled)
      reset_cache();
   // copy received vector, needed for lazy computation
   This::r = r;
   // copy a-priori statistics, needed for lazy computation (may be empty)
   This::app = app;
#ifndef NDEBUG
   if (app.size() == 0)
      std::cerr << "DEBUG (fba2): Empty APP table." << std::endl;
#endif
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_alpha(const array1d_t& sof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Alpha");
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // initialise array:
   alpha = real(0);
   // set initial drift distribution
   for (int x = -xmax; x <= xmax; x++)
      alpha[0][x] = real(sof_prior(xmax + x));
   // compute remaining matrix values
   for (int i = 1; i <= N; i++)
      {
      std::cerr << progress.update(i - 1, N);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (int x1 = -xmax; x1 <= xmax; x1++)
            if (alpha[i - 1][x1] > threshold)
               threshold = alpha[i - 1][x1];
         threshold *= th_inner;
         }
      // limits on insertions and deletions must be respected:
      //   x2-x1 <= n*I
      //   x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      //   x2-x1 <= dxmax
      //   x2-x1 >= -dxmax
      for (int x1 = -xmax; x1 <= xmax; x1++)
         {
         // ignore paths below a certain threshold
         if (thresholding && alpha[i - 1][x1] < threshold)
            continue;
         const int x2min = std::max(-xmax, dmin + x1);
         const int x2max = std::min(xmax, dmax + x1);
         for (int x2 = x2min; x2 <= x2max; x2++)
            for (int d = 0; d < q; d++)
               alpha[i][x2] += alpha[i - 1][x1] * get_gamma(d, i - 1, x1, x2
                     - x1);
         }
      // normalize if requested
      // TODO: refactor this out to a separate method
      if (norm)
         {
         real scale = 0;
         for (int x = -xmax; x <= xmax; x++)
            scale += alpha[i][x];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (int x = -xmax; x <= xmax; x++)
            alpha[i][x] *= scale;
         }
      }
   std::cerr << progress.update(N, N);
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_beta(const array1d_t& eof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Beta");
   // local flag for path thresholding
   const bool thresholding = (th_inner > real(0));
   // initialise array:
   beta = real(0);
   // set final drift distribution
   for (int x = -xmax; x <= xmax; x++)
      beta[N][x] = real(eof_prior(xmax + x));
   // compute remaining matrix values
   for (int i = N - 1; i >= 0; i--)
      {
      std::cerr << progress.update(N - 1 - i, N);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (int x2 = -xmax; x2 <= xmax; x2++)
            if (beta[i + 1][x2] > threshold)
               threshold = beta[i + 1][x2];
         threshold *= th_inner;
         }
      // limits on insertions and deletions must be respected:
      //   x2-x1 <= n*I
      //   x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      //   x2-x1 <= dxmax
      //   x2-x1 >= -dxmax
      for (int x2 = -xmax; x2 <= xmax; x2++)
         {
         // ignore paths below a certain threshold
         if (thresholding && beta[i + 1][x2] < threshold)
            continue;
         const int x1min = std::max(-xmax, x2 - dmax);
         const int x1max = std::min(xmax, x2 - dmin);
         for (int x1 = x1min; x1 <= x1max; x1++)
            for (int d = 0; d < q; d++)
               beta[i][x1] += beta[i + 1][x2] * get_gamma(d, i, x1, x2 - x1);
         }
      // normalize if requested
      // TODO: refactor this out to a separate method
      if (norm)
         {
         real scale = 0;
         for (int x = -xmax; x <= xmax; x++)
            scale += beta[i][x];
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (int x = -xmax; x <= xmax; x++)
            beta[i][x] *= scale;
         }
      }
   std::cerr << progress.update(N, N);
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_message_app(array1vr_t& ptable) const
   {
   assert(initialised);
   libbase::pacifier progress("FBA Results");
   // local flag for path thresholding
   const bool thresholding = (th_outer > real(0));
   // Initialise result vector (one sparse symbol per timestep)
   libbase::allocate(ptable, N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (int x1 = -xmax; x1 <= xmax; x1++)
            if (alpha[i][x1] > threshold)
               threshold = alpha[i][x1];
         threshold *= th_outer;
         }
      for (int d = 0; d < q; d++)
         {
         real p = 0;
         // limits on insertions and deletions must be respected:
         //   x2-x1 <= n*I
         //   x2-x1 >= -n
         // limits on introduced drift in this section:
         // (necessary for forward recursion on extracted segment)
         //   x2-x1 <= dxmax
         //   x2-x1 >= -dxmax
         for (int x1 = -xmax; x1 <= xmax; x1++)
            {
            // ignore paths below a certain threshold
            if (thresholding && alpha[i][x1] < threshold)
               continue;
            const int x2min = std::max(-xmax, dmin + x1);
            const int x2max = std::min(xmax, dmax + x1);
            for (int x2 = x2min; x2 <= x2max; x2++)
               p += alpha[i][x1] * get_gamma(d, i, x1, x2 - x1)
                     * beta[i + 1][x2];
            }
         ptable(i)(d) = p;
         }
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
#ifndef NDEBUG
   // show cache statistics
   std::cerr << "FBA Cache Usage: " << 100 * gamma_misses
         / double(cached.num_elements()) << "%" << std::endl;
   std::cerr << "FBA Cache Reuse: " << gamma_calls / double(gamma_misses * q)
         << "x" << std::endl;
#endif
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_state_app(array1r_t& ptable, const int i) const
   {
   assert(initialised);
   assert(i >= 0 && i <= N);
   // compute posterior probabilities for given index
   ptable.init(2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      ptable(xmax + x) = alpha[i][x] * beta[i][x];
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::work_results(array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post) const
   {
   // compute APPs of message
   work_message_app(ptable);
   // compute APPs of sof/eof state values
   work_state_app(sof_post, 0);
   work_state_app(eof_post, N);
   }

// User procedures

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
template <class real, class sig, bool norm>
void fba2<real, sig, norm>::decode(libcomm::instrumented& collector,
      const array1s_t& r, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post, const int offset)
   {
   // Initialise memory if necessary
   if (!initialised)
      allocate();
   // Validate sizes and offset
   const int tau = N * n;
   assertalways(offset == xmax);
   assertalways(r.size() == tau + 2 * xmax);
   assertalways(sof_prior.size() == 2 * xmax + 1);
   assertalways(eof_prior.size() == 2 * xmax + 1);
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
   collector.add_timer(sizeof(real) * gamma.num_elements(), "m_gamma");

#if DEBUG>=3
   std::cerr << "r = " << r << std::endl;
   if (cache_enabled)
      {
      std::cerr << "gamma = " << std::endl;
      // gamma has indices (d,i,x,deltax) where:
      //    d in [0, q-1], i in [0, N-1], x in [-xmax, xmax], and
      //    deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
      for (int i = 0; i < N; i++)
         {
         std::cerr << "i = " << i << ":" << std::endl;
         for (int d = 0; d < q; d++)
            {
            std::cerr << "d = " << d << ":" << std::endl;
            for (int x = -xmax; x <= xmax; x++)
               {
               for (int deltax = dmin; deltax <= dmax; deltax++)
               std::cerr << '\t' << gamma[d][i][x][deltax];
               std::cerr << std::endl;
               }
            }
         }
      }
   std::cerr << "alpha = " << alpha << std::endl;
   std::cerr << "beta = " << beta << std::endl;
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "norm = " << norm << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
// specialist arithmetic
template class fba2<logrealfast, bool, false> ;

// no-normalization, for debugging
template class fba2<float, bool, false> ;
template class fba2<double, bool, false> ;
// normalized, for normal use
template class fba2<float, bool, true> ;
template class fba2<double, bool, true> ;

} // end namespace
