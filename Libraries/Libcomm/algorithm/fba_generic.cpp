/*!
 * \file
 * $Id$
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

#include "fba_generic.h"
#include "pacifier.h"
#include "cputimer.h"
#include "field_utils.h"
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

/*! \brief Memory allocator for working matrices
 */
template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::allocate()
   {
   // Allocate required size
   // alpha needs indices (j,y) where j in [0, N] and y in [-xmax, xmax]
   // beta needs indices (j,y) where j in [0, N] and y in [-xmax, xmax]
   typedef boost::multi_array_types::extent_range range;
   alpha.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);
   beta.resize(boost::extents[N + 1][range(-xmax, xmax + 1)]);
   // flag the state of the arrays
   initialised = true;

   // if this is not the first time, skip the rest
   static bool first_time = true;
   if (!first_time)
      return;
   first_time = false;

#ifndef NDEBUG
   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const int prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(real)
         * (alpha.num_elements() + beta.num_elements());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
#endif

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "alpha = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << alpha.num_elements() << std::endl;
   std::cerr << "beta = " << N + 1 << "x" << 2 * xmax + 1 << " = "
   << beta.num_elements() << std::endl;
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::free()
   {
   alpha.resize(boost::extents[0][0]);
   beta.resize(boost::extents[0][0]);
   // flag the state of the arrays
   initialised = false;
   }

// Internal procedures

template <class sig, class real, class real2>
real fba_generic<sig, real, real2>::get_scale(const array2r_t& metric, int row,
      int col_min, int col_max)
   {
   real scale = 0;
   for (int col = col_min; col <= col_max; col++)
      scale += metric[row][col];
   assertalways(scale > real(0));
   scale = real(1) / scale;
   return scale;
   }

template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::normalize(array2r_t& metric, int row,
      int col_min, int col_max)
   {
   // determine the scale factor to use (each block has to do this)
   const real scale = get_scale(metric, row, col_min, col_max);
   // scale all results
   for (int col = col_min; col <= col_max; col++)
      metric[row][col] *= scale;
   }

template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::work_alpha(const array1s_t& r,
      const array1vd_t& app, const array1d_t& sof_prior)
   {
   assert(initialised);
   libbase::pacifier progress("FBA Alpha");
   // initialise array:
   alpha = real(0);
   // set initial drift distribution
   for (int x = -xmax; x <= xmax; x++)
      alpha[0][x] = real(sof_prior(xmax + x));
   // normalize if requested
   if (norm)
      normalize_alpha(0);
   // compute remaining matrix values
   for (int i = 1; i <= N; i++)
      {
      std::cerr << progress.update(i - 1, N);
      // compute partial result
      for (int x1 = -xmax; x1 <= xmax; x1++)
         {
         // cache previous alpha value in a register
         const real prev_alpha = alpha[i - 1][x1];
         // limits on insertions and deletions must be respected:
         //   x2-x1 <= I
         //   x2-x1 >= -1
         const int x2min = std::max(-xmax, x1 - 1);
         const int x2max = std::min(xmax, x1 + I);
         for (int x2 = x2min; x2 <= x2max; x2++)
            {
            // determine received segment to extract
            const int start = xmax + (i - 1) + x1;
            const int length = 1 + (x2 - x1);
            const array1s_t& r_segment = r.extract(start, length);
            // NOTE: we're repeating the loop on x2, so we need to increment this
            real this_alpha = alpha[i][x2];
            for (int d = 0; d < field_utils<sig>::elements(); d++)
               {
               real temp = prev_alpha;
               // call receiver method
               temp *= computer.receive(d, r_segment);
               // apply priors if applicable
               if (app.size() > 0)
                  temp *= real(app(i - 1)(d));
               this_alpha += temp;
               }
            alpha[i][x2] = this_alpha;
            }
         }
      // normalize if requested
      if (norm)
         normalize_alpha(i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   std::cerr << "alpha = " << alpha << std::endl;
#endif
   }

template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::work_beta(const array1s_t& r,
      const array1vd_t& app, const array1d_t& eof_prior)
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
   if (norm)
      normalize_beta(N);
   // compute remaining matrix values
   for (int i = N - 1; i >= 0; i--)
      {
      std::cerr << progress.update(N - 1 - i, N);
      // compute partial result
      for (int x1 = -xmax; x1 <= xmax; x1++)
         {
         real this_beta = 0;
         // limits on insertions and deletions must be respected:
         //   x2-x1 <= I
         //   x2-x1 >= -1
         const int x2min = std::max(-xmax, x1 - 1);
         const int x2max = std::min(xmax, x1 + I);
         for (int x2 = x2min; x2 <= x2max; x2++)
            {
            // determine received segment to extract
            const int start = xmax + i + x1;
            const int length = 1 + (x2 - x1);
            const array1s_t& r_segment = r.extract(start, length);
            // cache next beta value in a register
            const real next_beta = beta[i + 1][x2];
            for (int d = 0; d < field_utils<sig>::elements(); d++)
               {
               real temp = next_beta;
               // call receiver method
               temp *= computer.receive(d, r_segment);
               // apply priors if applicable
               if (app.size() > 0)
                  temp *= real(app(i)(d));
               this_beta += temp;
               }
            }
         beta[i][x1] = this_beta;
         }
      // normalize if requested
      if (norm)
         normalize_beta(i);
      }
   std::cerr << progress.update(N, N);
#if DEBUG>=3
   std::cerr << "beta = " << beta << std::endl;
#endif
   }

template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::work_message_app(const array1s_t& r,
      const array1vd_t& app, array1vr_t& ptable) const
   {
   libbase::pacifier progress("FBA Results");
   // Initialise result vector (one symbol per timestep)
   libbase::allocate(ptable, N, field_utils<sig>::elements());
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // compute partial result
      for (int d = 0; d < field_utils<sig>::elements(); d++)
         {
         // initialize result holder
         real p = 0;
         for (int x1 = -xmax; x1 <= xmax; x1++)
            {
            // cache this alpha value in a register
            const real this_alpha = alpha[i][x1];
            // limits on insertions and deletions must be respected:
            //   x2-x1 <= I
            //   x2-x1 >= -1
            const int x2min = std::max(-xmax, x1 - 1);
            const int x2max = std::min(xmax, x1 + I);
            for (int x2 = x2min; x2 <= x2max; x2++)
               {
               // determine received segment to extract
               const int start = xmax + i + x1;
               const int length = 1 + (x2 - x1);
               const array1s_t& r_segment = r.extract(start, length);
               // accumulate result
               real temp = this_alpha;
               temp *= beta[i + 1][x2];
               // call receiver method
               temp *= computer.receive(d, r_segment);
               // apply priors if applicable
               if (app.size() > 0)
                  temp *= real(app(i)(d));
               p += temp;
               }
            }
         // store result
         ptable(i)(d) = p;
         }
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
#if DEBUG>=3
   std::cerr << "ptable = " << ptable << std::endl;
#endif
   }

template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::work_state_app(array1r_t& ptable,
      const int i) const
   {
   assert(i >= 0 && i <= N);
   // compute posterior probabilities for given index
   ptable.init(2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      ptable(xmax + x) = alpha[i][x] * beta[i][x];
#if DEBUG>=3
   std::cerr << "state_post(" << i << ") = " << ptable << std::endl;
#endif
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
template <class sig, class real, class real2>
void fba_generic<sig, real, real2>::decode(libcomm::instrumented& collector,
      const array1s_t& r, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post, const int offset)
   {
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << N << std::endl;
   std::cerr << "I = " << I << std::endl;
   std::cerr << "xmax = " << xmax << std::endl;
   std::cerr << "norm = " << norm << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   // Initialise memory if necessary
   if (!initialised)
      allocate();
   // Validate sizes and offset
   assertalways(offset == xmax);
   assertalways(r.size() == N + 2 * xmax);
   assertalways(sof_prior.size() == 2 * xmax + 1);
   assertalways(eof_prior.size() == 2 * xmax + 1);
#if DEBUG>=3
   // show input data
   std::cerr << "r = " << r << std::endl;
   std::cerr << "app = " << app << std::endl;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
#endif

   // Alpha
   libbase::cputimer ta("t_alpha");
   work_alpha(r, app, sof_prior);
   collector.add_timer(ta);
   // Beta
   libbase::cputimer tb("t_beta");
   work_beta(r, app, eof_prior);
   collector.add_timer(tb);
   // Compute results
   libbase::cputimer tr("t_results");
   work_results(r, app, ptable, sof_post, eof_post);
   collector.add_timer(tr);

   // Add values for limits that depend on channel conditions
   collector.add_timer(I, "c_I");
   collector.add_timer(xmax, "c_xmax");
   // Add memory usage
   collector.add_timer(sizeof(real) * alpha.num_elements(), "m_alpha");
   collector.add_timer(sizeof(real) * beta.num_elements(), "m_beta");
   }

} // end namespace

#include "gf.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double)

// *** Instantiations for marker: bool and gf types only ***

#define INSTANTIATE(r, args) \
      template class fba_generic<BOOST_PP_SEQ_ENUM(args)> ;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE,
      (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
