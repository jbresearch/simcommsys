/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "bcjr.h"     

namespace libcomm {

using libbase::matrix;

// Initialization

/*!
   \brief   Creator for class 'bcjr'.
   \param   encoder     The finite state machine used to encode the source.
   \param   tau         The block length of decoder (including tail bits).
   \param   startatzero True if the trellis for the underlying code is known to start at state zero.
   \param   endatzero   True if the trellis for the underlying code is known to end at state zero.

   Note that if the trellis is not defined as starting or ending at zero, then it is assumed that
   all starting and ending states (respectively) are equiprobable.
*/
template <class real, class dbl> void bcjr<real,dbl>::init(fsm& encoder, const int tau, const bool startatzero, const bool endatzero, const bool circular)
   {
   if(tau < 1)
      {
      std::cerr << "FATAL ERROR (bcjr): MAP decoder block size too small (" << tau << ")\n";
      exit(1);
      }
    
   bcjr::tau = tau;

   // Initialise constants
   K = encoder.num_inputs();
   N = encoder.num_outputs();
   M = encoder.num_states();
   nu = endatzero ? encoder.mem_order() : 0;

   // initialise LUT's for state table
   // this must be done here or we will have to keep a copy of the encoder
   lut_X.init(M, K);
   lut_m.init(M, K);
   lut_i.init(M);
   for(int mdash=0; mdash<M; mdash++)
      {
      {
      for(int i=0; i<K; i++)
         {
         encoder.reset(mdash);
         int input = i;
         lut_X(mdash, i) = encoder.step(input);
         lut_m(mdash, i) = encoder.state();
         }
      }
      // we should not need the following if the trellis does not end at zero
      if(endatzero)
         {
         encoder.reset(mdash);
         int i = fsm::tail;
         int X = encoder.step(i);
         lut_X(mdash, i) = X;
         lut_m(mdash, i) = encoder.state();
         lut_i(mdash) = i;
         }
      }
   
   // memory is only allocated in the first decode call, so we have to store the
   // startatzero and endatzero conditions for the intialisation time
   bcjr::startatzero = startatzero;
   bcjr::endatzero = endatzero;
   // flag to denote circular decoding - note that this only really makes sense
   // when the start/end state are unknown, hence the assertion
   bcjr::circular = circular;
   assert(!circular || (!startatzero && !endatzero));
   // set flag as necessary
   initialised = false;
   }

// Reset start- and end-state probabilities

template <class real, class dbl> void bcjr<real,dbl>::reset()
   {
   if(!initialised)
      {
      allocate();
      return;
      }

   // initialise alpha and beta arrays    
   alpha(0, 0) = startatzero ? 1 : 1.0/double(M);
   beta(tau, 0) = endatzero ? 1 : 1.0/double(M);
   for(int m=1; m<M; m++)
      {
      alpha(0, m) = startatzero ? 0 : 1.0/double(M);
      beta(tau, m) = endatzero ? 0 : 1.0/double(M);
      }
   }

// Memory allocation

template <class real, class dbl> void bcjr<real,dbl>::allocate()
   {
   // to save space, gamma is defined from 0 to tau-1, rather than 1 to tau.
   // for this reason, gamma_t (and only gamma_t) is actually written gamma[t-1, ...
   alpha.init(tau+1, M);
   beta.init(tau+1, M);
   gamma.init(tau, M, K);

   // flag the state of the arrays
   initialised = true;

   // initialise alpha and beta arrays    
   reset();
   }


// Creation/Destruction routines

template <class real, class dbl> bcjr<real,dbl>::bcjr()
   {
   initialised = false;
   }

template <class real, class dbl> bcjr<real,dbl>::bcjr(fsm& encoder, const int tau, const bool startatzero, const bool endatzero, const bool circular)
   {
   init(encoder, tau, startatzero, endatzero, circular);
   }

template <class real, class dbl> bcjr<real,dbl>::~bcjr()
   {
   }


// Internal functions

//! State probability metric - lambda(t,m) = Pr{S(t)=m, Y[1..tau]}
template <class real, class dbl> real bcjr<real,dbl>::lambda(const int t, const int m)
   {
   return alpha(t, m) * beta(t, m);
   }
   
//! Transition probability metric - sigma(t,m,i) = Pr{S(t-1)=m, S(t)=m(m,i), Y[1..tau]}
template <class real, class dbl> real bcjr<real,dbl>::sigma(const int t, const int m, const int i)
   {
   int mdash = lut_m(m, i);
   return alpha(t-1, m) * gamma(t-1, m, i) * beta(t, mdash);
   }


// Internal procedures

/*!
   \brief   Computes the gamma matrix.
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X

   For all values of t in [1,tau], the gamma values are worked out as specified by the BCJR equation.

   \warning
   The line 'gamma(t-1, mdash, m) = R(t-1, X)' was changed to '+=' on 27 May 1998 to allow for
   the case (as in uncoded Tx) where trellis has parallel paths (more than one path starting at
   the same state and ending at the same state). It was then immediately changed back to '='
   because the BCJR algorithm cannot determine between two parallel paths anyway (algorithm is
   useless in such cases). Same applies to viterbi algorithm.
*/
template <class real, class dbl> void bcjr<real,dbl>::work_gamma(const matrix<dbl>& R)
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
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value)
                  i at time t

   For all values of t in [1,tau], the gamma values are worked out as specified by the BCJR equation.
   This function also makes use of the a priori probabilities associated with the input.
*/
template <class real, class dbl> void bcjr<real,dbl>::work_gamma(const matrix<dbl>& R, const matrix<dbl>& app)
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

   Alpha values only depend on the initial values (for t=0) and on the computed gamma values;
   the matrix is recursively computed. Initial alpha values are set in the creator and are never
   changed in the object's lifetime.
*/
template <class real, class dbl> void bcjr<real,dbl>::work_alpha()
   {
   // using the computed gamma values, work out all alpha values at time t
   for(int t=1; t<=tau; t++)
      {
      // first initialise the next set of alpha entries
      {
      for(int m=0; m<M; m++)
         alpha(t, m) = 0;
      }
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
      {
      for(int m=1; m<M; m++)
         if(alpha(t, m) > scale)
            scale = alpha(t, m);
      }
      {
      scale = real(1)/scale;
      for(int m=0; m<M; m++)
         alpha(t, m) *= scale;
      }
      }
   // when using a circular trellis, replace the start-state probabilities with the
   // end-state probabilities (scaling has already been performed
   if(circular)
      for(int m=0; m<M; m++)
         alpha(0, m) = alpha(tau, m);
   }

/*!
   \brief   Computes the beta matrix.

   Beta values only depend on the final values (for t=tau) and on the computed gamma values;
   the matrix is recursively computed. Final beta values are set in the creator and are never
   changed in the object's lifetime.
*/
template <class real, class dbl> void bcjr<real,dbl>::work_beta()
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
   // when using a circular trellis, replace the end-state probabilities with the
   // start-state probabilities (scaling has already been performed
   if(circular)
      for(int m=0; m<M; m++)
         beta(tau, m) = beta(0, m);
   }

/*!
   \brief   Computes the final results for the BCJR algorithm.
   \param   ri    ri(t-1, i) is the probability that we transmitted (input value) i at time t
   \param   ro    ro(t-1, X) is the probability that we transmitted (output value) X at time t

   Once we have worked out the gamma, alpha, and beta matrices, we are in a position to compute
   Py (the probability of having received the received sequence of modulation symbols). Next, we
   compute the results by doing the appropriate summations on sigma.

   \warning
   Initially, I used to work out the delta probability as:
      'delta = lambda(t-1, mdash)/Py * sigma(t, mdash, m)/Py'.
   I suspected this reasoning to be false, and am now working the delta value as:
      'delta = sigma(t, mdash, m)/Py'.
   This makes sense because the sigma values already take into account the probability of being in
   state mdash before the transition being considered (we care about the transition because this
   determines the input and output symbols represented).
*/
template <class real, class dbl> void bcjr<real,dbl>::work_results(matrix<dbl>& ri, matrix<dbl>& ro)
   {
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // initialise results
   ri = dbl(0);
   ro = dbl(0);
   // Work out final results
   for(int t=1; t<=tau; t++)
      for(int mdash=0; mdash<M; mdash++)        // for each possible state at time t-1
         for(int i=0; i<K; i++) // for each possible input, given the state we were in
            {
            int X = lut_X(mdash, i);
            dbl delta = sigma(t, mdash, i)/Py;
            ri(t-1, i) += delta;
            ro(t-1, X) += delta;
            }
   }

/*!
   \brief   Computes the final results for the BCJR algorithm (input statistics only).
   \param   ri    ri(t-1, i) is the probability that we transmitted (input value) i at time t

   Once we have worked out the gamma, alpha, and beta matrices, we are in a position to compute
   Py (the probability of having received the received sequence of modulation symbols). Next, we
   compute the results by doing the appropriate summations on sigma.
*/
template <class real, class dbl> void bcjr<real,dbl>::work_results(matrix<dbl>& ri)
   {
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // Work out final results
   for(int t=1; t<=tau; t++)
      for(int i=0; i<K; i++)    // for each possible input, given the state we were in
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
*/
template <class real, class dbl> void bcjr<real,dbl>::normalize(matrix<dbl>& r)
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
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X
   \param   ri    ri(t-1, i) is the a posteriori probability of having transmitted (input value)
                  i at time t (result)
   \param   ro    ro(t-1, X) = (result) a posteriori probability of having transmitted (output value)
                  X at time t (result)
*/
template <class real, class dbl> void bcjr<real,dbl>::decode(const matrix<dbl>& R, matrix<dbl>& ri, matrix<dbl>& ro)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // forward pass
   work_gamma(R);
   work_alpha();

   // backward pass
   work_beta();
   
   // Work out final results
   work_results(ri, ro);
   }

/*!
   \brief   Wrapping function for decoding a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value)
                  i at time t
   \param   ri    ri(t-1, i) is the a posteriori probability of having transmitted (input value)
                  i at time t (result)
   \param   ro    ro(t-1, X) = (result) a posteriori probability of having transmitted (output value)
                  X at time t (result)
*/
template <class real, class dbl> void bcjr<real,dbl>::decode(const matrix<dbl>& R, const matrix<dbl>& app, matrix<dbl>& ri, matrix<dbl>& ro)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // forward pass
   work_gamma(R, app);
   work_alpha();

   // backward pass
   work_beta();
   
   // Work out final results
   work_results(ri, ro);
   }

/*!
   \brief   Wrapping function for faster decoding of a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X
   \param   ri    ri(t-1, i) is the a posteriori probability of having transmitted (input value)
                  i at time t (result)
*/
template <class real, class dbl> void bcjr<real,dbl>::fdecode(const matrix<dbl>& R, matrix<dbl>& ri)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // forward pass
   work_gamma(R);
   work_alpha();

   // backward pass
   work_beta();
   
   // Work out final results
   work_results(ri);
   }

/*!
   \brief   Wrapping function for faster decoding of a block.
   \param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t,
                  having transmitted X
   \param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value)
                  i at time t
   \param   ri    ri(t-1, i) is the a posteriori probability of having transmitted (input value)
                  i at time t (result)
*/
template <class real, class dbl> void bcjr<real,dbl>::fdecode(const matrix<dbl>& R, const matrix<dbl>& app, matrix<dbl>& ri)
   {
   // initialise memory if necessary
   if(!initialised)
      allocate();

   // forward pass
   work_gamma(R, app);
   work_alpha();

   // backward pass
   work_beta();
   
   // Work out final results
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

using libbase::vcs;

#define VERSION 2.60

template class bcjr<double>;
template <> const vcs bcjr<double>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<double>)", VERSION);

template class bcjr<mpreal>;
template <> const vcs bcjr<mpreal>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<mpreal>)", VERSION);

template class bcjr<mpgnu>;
template <> const vcs bcjr<mpgnu>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<mpgnu>)", VERSION);

template class bcjr<logreal>;
template <> const vcs bcjr<logreal>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<logreal>)", VERSION);

template class bcjr<logrealfast>;
template <> const vcs bcjr<logrealfast>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<logrealfast>)", VERSION);

template class bcjr<logrealfast,logrealfast>;
template <> const vcs bcjr<logrealfast,logrealfast>::version = vcs("Bahl-Cocke-Jelinek-Raviv Algorithm module (bcjr<logrealfast,logrealfast>)", VERSION);

}; // end namespace
