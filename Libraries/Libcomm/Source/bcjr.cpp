/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bcjr.h"
#include <iomanip>

namespace libcomm {

// Initialization

/*!
   \brief   Creator for class 'bcjr'.
   \param   encoder     The finite state machine used to encode the source.
   \param   tau         The block length of decoder (including tail bits).

   \note If the trellis is not defined as starting or ending at zero, then it
         is assumed that all starting and ending states (respectively) are
         equiprobable.
*/
template <class real, class dbl>
void bcjr<real,dbl>::init(fsm& encoder, const int tau)
   {
   assertalways(tau > 0);
   bcjr::tau = tau;

   // Initialise constants
   K = encoder.num_inputs();
   N = encoder.num_outputs();
   M = encoder.num_states();

   // initialise LUT's for state table
   // this must be done here or we will have to keep a copy of the encoder
   lut_X.init(M, K);
   lut_m.init(M, K);
   for(int mdash=0; mdash<M; mdash++)
      for(int i=0; i<K; i++)
         {
         encoder.reset(mdash);
         int input = i;
         lut_X(mdash, i) = encoder.step(input);
         lut_m(mdash, i) = encoder.state();
         }

   // set flag as necessary
   initialised = false;
   }

// Get start- and end-state probabilities

template <class real, class dbl>
typename bcjr<real,dbl>::array1d_t bcjr<real,dbl>::getstart() const
   {
   array1d_t r(M);
   for(int m=0; m<M; m++)
      r(m) = beta(0, m);
   return r;
   }

template <class real, class dbl>
typename bcjr<real,dbl>::array1d_t bcjr<real,dbl>::getend() const
   {
   array1d_t r(M);
   for(int m=0; m<M; m++)
      r(m) = alpha(tau, m);
   return r;
   }

// Set start- and end-state probabilities - equiprobable

template <class real, class dbl>
void bcjr<real,dbl>::setstart()
   {
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      alpha(0, m) = 1.0/double(M);
   }

template <class real, class dbl>
void bcjr<real,dbl>::setend()
   {
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      beta(tau, m) = 1.0/double(M);
   }

// Set start- and end-state probabilities - known state

template <class real, class dbl>
void bcjr<real,dbl>::setstart(int state)
   {
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      alpha(0, m) = 0;
   alpha(0, state) = 1;
   }

template <class real, class dbl>
void bcjr<real,dbl>::setend(int state)
   {
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      beta(tau, m) = 0;
   beta(tau, state) = 1;
   }

// Set start- and end-state probabilities - direct

template <class real, class dbl>
void bcjr<real,dbl>::setstart(const array1d_t& p)
   {
   assert(p.size() == M);
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      alpha(0, m) = p(m);
   }

template <class real, class dbl>
void bcjr<real,dbl>::setend(const array1d_t& p)
   {
   assert(p.size() == M);
   if(!initialised)
      allocate();
   for(int m=0; m<M; m++)
      beta(tau, m) = p(m);
   }

// Internal methods

/*! \brief Memory allocator for working matrices
*/
template <class real, class dbl>
void bcjr<real,dbl>::allocate()
   {
   // to save space, gamma is defined from 0 to tau-1, rather than 1 to tau.
   // for this reason, gamma_t (and only gamma_t) is actually written gamma[t-1, ...
   alpha.init(tau+1, M);
   beta.init(tau+1, M);
   gamma.init(tau, M, K);
   // determine memory occupied and tell user
   std::ios::fmtflags flags = std::cerr.flags();
   std::cerr << "BCJR Memory Usage: " << std::fixed << std::setprecision(1);
   std::cerr << sizeof(real)*( alpha.size() + beta.size() + gamma.size()
                           )/double(1<<20) << "MB\n";
   std::cerr.setf(flags);
   // flag the state of the arrays
   initialised = true;
   }

/*! \brief State probability metric
   lambda(t,m) = Pr{S(t)=m, Y[1..tau]}
*/
template <class real, class dbl>
inline real bcjr<real,dbl>::lambda(const int t, const int m)
   {
   return alpha(t, m) * beta(t, m);
   }

/*! Transition probability metric
   sigma(t,m,i) = Pr{S(t-1)=m, S(t)=m(m,i), Y[1..tau]}
*/
template <class real, class dbl>
inline real bcjr<real,dbl>::sigma(const int t, const int m, const int i)
   {
   int mdash = lut_m(m, i);
   return alpha(t-1, m) * gamma(t-1, m, i) * beta(t, mdash);
   }

/*!
   \brief   Computes the gamma matrix.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X

   For all values of t in [1,tau], the gamma values are worked out as specified
   by the BCJR equation.
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_gamma(const array2d_t& R)
   {
   for(int t=1; t<=tau; t++)
      for(int mdash=0; mdash<M; mdash++)
         for(int i=0; i<K; i++)
            {
            int X = lut_X(mdash, i);
            gamma(t-1, mdash, i) = R(t-1, X);
            }
   }

/*!
   \brief   Computes the gamma matrix.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having
                  transmitted (input value) i at time t

   For all values of t in [1,tau], the gamma values are worked out as specified
   by the BCJR equation. This function also makes use of the a priori
   probabilities associated with the input.
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_gamma(const array2d_t& R, const array2d_t& app)
   {
   for(int t=1; t<=tau; t++)
      for(int mdash=0; mdash<M; mdash++)
         for(int i=0; i<K; i++)
            {
            int X = lut_X(mdash, i);
            gamma(t-1, mdash, i) = R(t-1, X) * app(t-1, i);
            }
   }

/*!
   \brief   Computes the alpha matrix.

   Alpha values only depend on the initial values (for t=0) and on the computed
   gamma values; the matrix is recursively computed. Initial alpha values are
   set in the creator and are never changed in the object's lifetime.

   \note Metrics are normalized using a variation of Matt Valenti's CML Theory
         slides; this was initially an attempt at solving the numerical range
         problems in multiple (sets>2) Turbo codes.
         Rather than dividing by the value for the first symbol, we determine
         the maximum value over all symbols and divide by that. This avoids
         problems when the metric for the first symbol is very small.
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_alpha()
   {
   // using the computed gamma values, work out all alpha values at time t
   for(int t=1; t<=tau; t++)
      {
      // first initialise the next set of alpha entries
      for(int m=0; m<M; m++)
         alpha(t, m) = 0;
      // now start computing the summations
      // tail conditions are automatically handled by zeros in the gamma matrix
      for(int mdash=0; mdash<M; mdash++)
         for(int i=0; i<K; i++)
            {
            int m = lut_m(mdash, i);
            alpha(t, m) += alpha(t-1, mdash) * gamma(t-1, mdash, i);
            }
      // normalize
      real scale = alpha(t, 0);
      for(int m=1; m<M; m++)
         if(alpha(t, m) > scale)
            scale = alpha(t, m);
      scale = real(1)/scale;
      for(int m=0; m<M; m++)
         alpha(t, m) *= scale;
      }
   }

/*!
   \brief   Computes the beta matrix.

   Beta values only depend on the final values (for t=tau) and on the computed
   gamma values; the matrix is recursively computed. Final beta values are set
   in the creator and are never changed in the object's lifetime.

   \sa See notes for work_alpha()
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_beta()
   {
   // evaluate all beta values
   for(int t=tau-1; t>=0; t--)
      {
      for(int m=0; m<M; m++)
         {
         beta(t, m) = 0;
         for(int i=0; i<K; i++)
            {
            int mdash = lut_m(m, i);
            beta(t, m) += beta(t+1, mdash) * gamma(t, m, i);
            }
         }
      // normalize
      real scale = beta(t, 0);
      for(int m=1; m<M; m++)
         if(beta(t, m) > scale)
            scale = beta(t, m);
      scale = real(1)/scale;
      for(int m=0; m<M; m++)
         beta(t, m) *= scale;
      }
   }

/*!
   \brief   Computes the final results for the BCJR algorithm.
   \param   ri    ri(t-1, i) is the probability that we transmitted
                  (input value) i at time t
   \param   ro    ro(t-1, X) is the probability that we transmitted
                  (output value) X at time t

   Once we have worked out the gamma, alpha, and beta matrices, we are in a
   position to compute Py (the probability of having received the received
   sequence of modulation symbols). Next, we compute the results by doing
   the appropriate summations on sigma.

   \warning Initially, I used to work out the delta probability as:
               delta = lambda(t-1, mdash)/Py * sigma(t, mdash, m)/Py
            I suspected this reasoning to be false, and am now working the
            delta value as:
               delta = sigma(t, mdash, m)/Py
            This makes sense because the sigma values already take into account
            the probability of being in state mdash before the transition being
            considered (we care about the transition because this determines
            the input and output symbols represented).
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_results(array2d_t& ri, array2d_t& ro)
   {
   // Initialize results vectors
   ri.init(tau, K);
   ro.init(tau, N);
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // initialise results
   ri = dbl(0);
   ro = dbl(0);
   // Work out final results
   for(int t=1; t<=tau; t++)
      for(int mdash=0; mdash<M; mdash++) // for each possible state at time t-1
         for(int i=0; i<K; i++) // for each possible input, given present state
            {
            int X = lut_X(mdash, i);
            dbl delta = sigma(t, mdash, i)/Py;
            ri(t-1, i) += delta;
            ro(t-1, X) += delta;
            }
   }

/*!
   \brief   Computes the final results for the BCJR algorithm (input statistics
            only).
   \param   ri    ri(t-1, i) is the probability that we transmitted
                  (input value) i at time t

   Once we have worked out the gamma, alpha, and beta matrices, we are in a
   position to compute Py (the probability of having received the received
   sequence of modulation symbols). Next, we compute the results by doing the
   appropriate summations on sigma.
*/
template <class real, class dbl>
void bcjr<real,dbl>::work_results(array2d_t& ri)
   {
   // Initialize results vector
   ri.init(tau, K);
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // Work out final results
   for(int t=1; t<=tau; t++)
      for(int i=0; i<K; i++) // for each possible input, given present state
         {
         // initialise results (we divide on every increment as division is faster
         // than addition for logreal - this makes other representations slower)
         dbl delta = 0;
         for(int mdash=0; mdash<M; mdash++)     // for each possible state at time t-1
            delta += dbl(sigma(t, mdash, i)/Py);
         // copy results into their final place
         ri(t-1, i) = delta;
         }
   }

// Internal helper functions

/*!
   \brief   Function to normalize results vectors
   \param   r     matrix with results - first index represents time-step

   This function is provided for derived classes to use; rather than
   normalizing the a-priori and a-posteriori probabilities in this class, it
   is up to derived classes to decide when that should be done. The reason
   behind this is that this class should not be responsible for its inputs,
   but whoever is providing them is.
*/
template <class real, class dbl>
void bcjr<real,dbl>::normalize(array2d_t& r)
   {
   for(int t=0; t<r.xsize(); t++)
      {
      dbl scale = r(t,0);
      for(int i=1; i<r.ysize(); i++)
         if(r(t,i) > scale)
            scale = r(t,i);
      if(scale > dbl(0))
         {
         scale = dbl(1)/scale;
         for(int i=0; i<r.ysize(); i++)
            r(t,i) *= scale;
         }
      }
   }

// User procedures

/*!
   \brief   Wrapping function for decoding a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X
   \param   ri    ri(t-1, i) is the a posteriori probability of having
                  transmitted (input value) i at time t (result)
   \param   ro    ro(t-1, X) = (result) a posteriori probability of having
                  transmitted (output value) X at time t (result)
*/
template <class real, class dbl>
void bcjr<real,dbl>::decode(const array2d_t& R, array2d_t& ri, array2d_t& ro)
   {
   assert(initialised);
   work_gamma(R);
   work_alpha();
   work_beta();
   work_results(ri, ro);
   }

/*!
   \brief   Wrapping function for decoding a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having
                  transmitted (input value) i at time t
   \param   ri    ri(t-1, i) is the a posteriori probability of having
                  transmitted (input value) i at time t (result)
   \param   ro    ro(t-1, X) = (result) a posteriori probability of having
                  transmitted (output value) X at time t (result)

   This is the same as the regular decoder, but does not produce a posteriori
   statistics on the decoder's output.
*/
template <class real, class dbl>
void bcjr<real,dbl>::decode(const array2d_t& R, const array2d_t& app, array2d_t& ri, array2d_t& ro)
   {
   assert(initialised);
   work_gamma(R, app);
   work_alpha();
   work_beta();
   work_results(ri, ro);
   }

/*!
   \brief   Wrapping function for faster decoding of a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X
   \param   ri    ri(t-1, i) is the a posteriori probability of having
                  transmitted (input value) i at time t (result)

   This is the same as the regular decoder, but does not produce a posteriori
   statistics on the decoder's output.
*/
template <class real, class dbl>
void bcjr<real,dbl>::fdecode(const array2d_t& R, array2d_t& ri)
   {
   assert(initialised);
   work_gamma(R);
   work_alpha();
   work_beta();
   work_results(ri);
   }

/*!
   \brief   Wrapping function for faster decoding of a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we
                  received" at time t, having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having
                  transmitted (input value) i at time t
   \param   ri    ri(t-1, i) is the a posteriori probability of having
                  transmitted (input value) i at time t (result)
*/
template <class real, class dbl>
void bcjr<real,dbl>::fdecode(const array2d_t& R, const array2d_t& app, array2d_t& ri)
   {
   assert(initialised);
   work_gamma(R, app);
   work_alpha();
   work_beta();
   work_results(ri);
   }

}; // end namespace

// Explicit Realizations

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

template class bcjr<double>;
template class bcjr<mpreal>;
template class bcjr<mpgnu>;
template class bcjr<logreal>;
template class bcjr<logrealfast>;
template class bcjr<logrealfast,logrealfast>;

}; // end namespace
