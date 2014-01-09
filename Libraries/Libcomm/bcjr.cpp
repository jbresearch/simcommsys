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

#include "bcjr.h"
#include <iomanip>

namespace libcomm {

// Initialization

/*!
 * \brief   Creator for class 'bcjr'.
 * \param   encoder     The finite state machine used to encode the source.
 * \param   tau         The block length of decoder (including tail bits).
 *
 * \note If the trellis is not defined as starting or ending at zero, then it
 * is assumed that all starting and ending states (respectively) are
 * equiprobable.
 *
 * \note Instead of keeping a copy of the encoder, we compute the state
 * transition and output tables and keep a copy of those.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::init(fsm& encoder, const int tau)
   {
   assertalways(tau > 0);
   bcjr::tau = tau;

   // Initialise constants
   K = encoder.num_input_combinations();
   N = encoder.num_output_combinations();
   M = encoder.num_states();

   // initialise LUT's for state table
   lut_X.init(M, K);
   lut_m.init(M, K);
   for (int mdash = 0; mdash < M; mdash++)
      for (int i = 0; i < K; i++)
         {
         array1i_t mdash_v = encoder.convert_state(mdash);
         encoder.reset(mdash_v);
         array1i_t input = encoder.convert_input(i);
         lut_X(mdash, i) = encoder.convert_output(encoder.step(input));
         assert(lut_X(mdash, i) >= 0 && lut_X(mdash, i) < N);
         lut_m(mdash, i) = encoder.convert_state(encoder.state());
         assert(lut_m(mdash, i) >= 0 && lut_m(mdash, i) < M);
         }

   // set flag as necessary
   initialised = false;
   }

// Get start- and end-state probabilities

template <class real, class dbl, bool norm>
typename bcjr<real, dbl, norm>::array1d_t bcjr<real, dbl, norm>::getstart() const
   {
   array1d_t r(M);
   for (int m = 0; m < M; m++)
      r(m) = dbl(beta(0, m));
   return r;
   }

template <class real, class dbl, bool norm>
typename bcjr<real, dbl, norm>::array1d_t bcjr<real, dbl, norm>::getend() const
   {
   array1d_t r(M);
   for (int m = 0; m < M; m++)
      r(m) = dbl(alpha(tau, m));
   return r;
   }

// Set start- and end-state probabilities - equiprobable

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setstart()
   {
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      alpha(0, m) = real(1.0 / M);
   }

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setend()
   {
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      beta(tau, m) = real(1.0 / M);
   }

// Set start- and end-state probabilities - known state

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setstart(int state)
   {
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      alpha(0, m) = real(0);
   alpha(0, state) = real(1);
   }

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setend(int state)
   {
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      beta(tau, m) = real(0);
   beta(tau, state) = real(1);
   }

// Set start- and end-state probabilities - direct

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setstart(const array1d_t& p)
   {
   assert(p.size() == M);
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      alpha(0, m) = real(p(m));
   }

template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::setend(const array1d_t& p)
   {
   assert(p.size() == M);
   if (!initialised)
      allocate();
   for (int m = 0; m < M; m++)
      beta(tau, m) = real(p(m));
   }

// Internal methods

/*! \brief Memory allocator for working matrices
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::allocate()
   {
   // to save space, gamma is defined from 0 to tau-1, rather than 1 to tau.
   // for this reason, gamma_t (and only gamma_t) is actually written gamma[t-1, ...
   alpha.init(tau + 1, M);
   beta.init(tau + 1, M);
   gamma.init(tau, M, K);
   // flag the state of the arrays
   initialised = true;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const std::streamsize prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(real) * (alpha.size() + beta.size()
         + gamma.size());
   std::cerr << "BCJR Memory Usage: " << bytes_used / double(1 << 20)
         << "MiB" << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
   }

/*! \brief State probability metric
 * lambda(t,m) = Pr{S(t)=m, Y[1..tau]}
 */
template <class real, class dbl, bool norm>
inline real bcjr<real, dbl, norm>::lambda(const int t, const int m)
   {
   return alpha(t, m) * beta(t, m);
   }

/*! \brief Transition probability metric
 * sigma(t,m,i) = Pr{S(t-1)=m, S(t)=m(m,i), Y[1..tau]}
 */
template <class real, class dbl, bool norm>
inline real bcjr<real, dbl, norm>::sigma(const int t, const int m, const int i)
   {
   int mdash = lut_m(m, i);
   return alpha(t - 1, m) * gamma(t - 1, m, i) * beta(t, mdash);
   }

/*!
 * \brief   Computes the gamma matrix.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 *
 * For all values of t in [1,tau], the gamma values are worked out as specified
 * by the BCJR equation.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_gamma(const array2d_t& R)
   {
   for (int t = 1; t <= tau; t++)
      for (int mdash = 0; mdash < M; mdash++)
         for (int i = 0; i < K; i++)
            {
            int X = lut_X(mdash, i);
            gamma(t - 1, mdash, i) = real(R(t - 1, X));
            }
   }

/*!
 * \brief   Computes the gamma matrix.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 * \param   app   app(t-1, i) is the 'a priori' probability of having
 * transmitted (input value) i at time t
 *
 * For all values of t in [1,tau], the gamma values are worked out as specified
 * by the BCJR equation. This function also makes use of the a priori
 * probabilities associated with the input.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_gamma(const array2d_t& R, const array2d_t& app)
   {
   for (int t = 1; t <= tau; t++)
      for (int mdash = 0; mdash < M; mdash++)
         for (int i = 0; i < K; i++)
            {
            int X = lut_X(mdash, i);
            gamma(t - 1, mdash, i) = real(R(t - 1, X) * app(t - 1, i));
            }
   }

/*!
 * \brief   Computes the alpha matrix.
 *
 * Alpha values only depend on the initial values (for t=0) and on the computed
 * gamma values; the matrix is recursively computed. Initial alpha values are
 * set in the creator and are never changed in the object's lifetime.
 *
 * \note Metrics are normalized using a variation of Matt Valenti's CML Theory
 * slides; this was initially an attempt at solving the numerical range
 * problems in multiple (sets>2) Turbo codes.
 * Rather than dividing by the value for the first symbol, we determine
 * the maximum value over all symbols and divide by that. This avoids
 * problems when the metric for the first symbol is very small.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_alpha()
   {
   // using the computed gamma values, work out all alpha values at time t
   for (int t = 1; t <= tau; t++)
      {
      // first initialise the next set of alpha entries
      for (int m = 0; m < M; m++)
         alpha(t, m) = 0;
      // now start computing the summations
      // tail conditions are automatically handled by zeros in the gamma matrix
      for (int mdash = 0; mdash < M; mdash++)
         for (int i = 0; i < K; i++)
            {
            int m = lut_m(mdash, i);
            alpha(t, m) += alpha(t - 1, mdash) * gamma(t - 1, mdash, i);
            }
      // normalize
      if (norm)
         {
         real scale = alpha(t, 0);
         for (int m = 1; m < M; m++)
            scale += alpha(t, m);
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (int m = 0; m < M; m++)
            alpha(t, m) *= scale;
         }
      }
   }

/*!
 * \brief   Computes the beta matrix.
 *
 * Beta values only depend on the final values (for t=tau) and on the computed
 * gamma values; the matrix is recursively computed. Final beta values are set
 * in the creator and are never changed in the object's lifetime.
 *
 * \sa See notes for work_alpha()
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_beta()
   {
   // evaluate all beta values
   for (int t = tau - 1; t >= 0; t--)
      {
      for (int m = 0; m < M; m++)
         {
         beta(t, m) = 0;
         for (int i = 0; i < K; i++)
            {
            int mdash = lut_m(m, i);
            beta(t, m) += beta(t + 1, mdash) * gamma(t, m, i);
            }
         }
      // normalize
      if (norm)
         {
         real scale = beta(t, 0);
         for (int m = 1; m < M; m++)
            scale += beta(t, m);
         assertalways(scale > real(0));
         scale = real(1) / scale;
         for (int m = 0; m < M; m++)
            beta(t, m) *= scale;
         }
      }
   }

/*!
 * \brief   Computes the final results for the BCJR algorithm.
 * \param   ri    ri(t-1, i) is the probability that we transmitted
 * (input value) i at time t
 * \param   ro    ro(t-1, X) is the probability that we transmitted
 * (output value) X at time t
 *
 * Once we have worked out the gamma, alpha, and beta matrices, we are in a
 * position to compute Py (the probability of having received the received
 * sequence of modulation symbols). Next, we compute the results by doing
 * the appropriate summations on sigma.
 *
 * \warning Initially, I used to work out the delta probability as:
 * delta = lambda(t-1, mdash)/Py * sigma(t, mdash, m)/Py
 * I suspected this reasoning to be false, and am now working the
 * delta value as:
 * delta = sigma(t, mdash, m)/Py
 * This makes sense because the sigma values already take into account
 * the probability of being in state mdash before the transition being
 * considered (we care about the transition because this determines
 * the input and output symbols represented).
 *
 * \todo Update according to the changes in work_results(ri)
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_results(array2d_t& ri, array2d_t& ro)
   {
   // Initialize results vectors
   ri.init(tau, K);
   ro.init(tau, N);
   // Compute probability of received sequence
   real Py = 0;
   for (int mdash = 0; mdash < M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // initialise results
   ri = dbl(0);
   ro = dbl(0);
   // Work out final results
   for (int t = 1; t <= tau; t++)
      for (int mdash = 0; mdash < M; mdash++) // for each possible state at time t-1
         for (int i = 0; i < K; i++) // for each possible input, given present state
            {
            int X = lut_X(mdash, i);
            dbl delta = dbl(sigma(t, mdash, i) / Py);
            ri(t - 1, i) += delta;
            ro(t - 1, X) += delta;
            }
   }

/*!
 * \brief   Computes the final results for the BCJR algorithm (input only).
 * \param   ri    ri(t-1, i) is the probability that we transmitted
 * (input value) i at time t
 *
 * Once we have worked out the gamma, alpha, and beta matrices, we are in a
 * position to compute Py (the probability of having received the received
 * sequence of modulation symbols). Next, we compute the results by doing the
 * appropriate summations on sigma.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::work_results(array2d_t& ri)
   {
   // Initialize results vector
   ri.init(tau, K);
   // Compute probability of received sequence
   real Py = 0;
   for (int mdash = 0; mdash < M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // Work out final results
   for (int t = 1; t <= tau; t++)
      for (int i = 0; i < K; i++) // for each possible input, given present state
         {
         // compute results
         real delta = 0;
         for (int mdash = 0; mdash < M; mdash++) // for each possible state at time t-1
            delta += sigma(t, mdash, i);
         // copy results into their final place
         ri(t - 1, i) = dbl(delta / Py);
         }
   }

// Internal helper functions

/*!
 * \brief   Function to normalize results vectors
 * \param   r     matrix with results - first index represents time-step
 *
 * This function is provided for derived classes to use; rather than
 * normalizing the a-priori and a-posteriori probabilities in this class, it
 * is up to derived classes to decide when that should be done. The reason
 * behind this is that this class should not be responsible for its inputs,
 * but whoever is providing them is.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::normalize(array2d_t& r)
   {
   for (int t = 0; t < r.size().rows(); t++)
      {
      dbl scale = r(t, 0);
      for (int i = 1; i < r.size().cols(); i++)
         scale += r(t, i);
      assertalways(scale > dbl(0));
      scale = dbl(1) / scale;
      for (int i = 0; i < r.size().cols(); i++)
         {
         r(t, i) *= scale;
         // TODO: replace with clipping
         //assert(r(t, i) > dbl(0));
         }
      }
   }

// User procedures

/*!
 * \brief   Wrapping function for decoding a block.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 * \param   ri    ri(t-1, i) is the a posteriori probability of having
 * transmitted (input value) i at time t (result)
 * \param   ro    ro(t-1, X) = (result) a posteriori probability of having
 * transmitted (output value) X at time t (result)
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::decode(const array2d_t& R, array2d_t& ri,
      array2d_t& ro)
   {
   assert(initialised);
   work_gamma(R);
   work_alpha();
   work_beta();
   work_results(ri, ro);
   }

/*!
 * \brief   Wrapping function for decoding a block.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 * \param   app   app(t-1, i) is the 'a priori' probability of having
 * transmitted (input value) i at time t
 * \param   ri    ri(t-1, i) is the a posteriori probability of having
 * transmitted (input value) i at time t (result)
 * \param   ro    ro(t-1, X) = (result) a posteriori probability of having
 * transmitted (output value) X at time t (result)
 *
 * This is the same as the regular decoder, but does not produce a posteriori
 * statistics on the decoder's output.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::decode(const array2d_t& R, const array2d_t& app,
      array2d_t& ri, array2d_t& ro)
   {
   assert(initialised);
   work_gamma(R, app);
   work_alpha();
   work_beta();
   work_results(ri, ro);
   }

/*!
 * \brief   Wrapping function for faster decoding of a block.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 * \param   ri    ri(t-1, i) is the a posteriori probability of having
 * transmitted (input value) i at time t (result)
 *
 * This is the same as the regular decoder, but does not produce a posteriori
 * statistics on the decoder's output.
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::fdecode(const array2d_t& R, array2d_t& ri)
   {
   assert(initialised);
   work_gamma(R);
   work_alpha();
   work_beta();
   work_results(ri);
   }

/*!
 * \brief   Wrapping function for faster decoding of a block.
 * \param   R     R(t-1, X) is the probability of receiving "whatever we
 * received" at time t, having transmitted X
 * \param   app   app(t-1, i) is the 'a priori' probability of having
 * transmitted (input value) i at time t
 * \param   ri    ri(t-1, i) is the a posteriori probability of having
 * transmitted (input value) i at time t (result)
 */
template <class real, class dbl, bool norm>
void bcjr<real, dbl, norm>::fdecode(const array2d_t& R, const array2d_t& app,
      array2d_t& ri)
   {
   assert(initialised);
   work_gamma(R, app);
   work_alpha();
   work_beta();
   work_results(ri);
   }

} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define TF_SEQ \
   (false)(true)
#define REAL1_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)
#define REAL2_TYPE_SEQ \
   (float)(double) \
   (logrealfast)

#define INSTANTIATE(r, args) \
      template class bcjr<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (REAL1_TYPE_SEQ)(REAL2_TYPE_SEQ)(TF_SEQ))

} // end namespace
