#ifndef __bcjr_h
#define __bcjr_h

#include "config.h"
#include "vcs.h"
#include "sigspace.h"
#include "matrix.h"
#include "vector.h"
#include "fsm.h"
#include <iostream.h>
#include <fstream.h>
#include <math.h>

extern const vcs bcjr_version;

/*!
\brief   Class implementing the BCJR decoding algorithm.
\author  Johann Briffa
\date    4 March 1999
\version 2.30

  All internal metrics are held as type 'real', which is user-defined. This allows internal working
  at any required level of accuracy. This is required because the internal matrics have a very wide
  dynamic range, which increases exponentially with block size 'tau'. Actually, the required range
  is within [1,0), but very large exponents are required.

  Version 2.00 (26 Feb 1999)
  updated by redefining the gamma matrix to contain only the connected set of mdash,m (reducing memory
  by a factor of M/K) and by computing alpha and beta accordingly (a speedup of M/K).

  Version 2.10 (4 Mar 1999)
  memory is only allocated in the first call to "decode". This is more efficient for the MPI strategy
  with a root server which only collects results.

  Version 2.20 (5 Mar 1999)
  modified the definition of nu (defined it as 0 for endatzero==false), in order to work out the gamma
  matrix correctly. Note that all previous results are invalid for endatzero==false!

  Version 2.30 (8 Mar 1999)
  added a faster decode routine (fdecode) which does not produce a posteriori statistics on the
  decoder's output.

  Version 2.31 (8 Mar 1999)
  optimised fdecode (reordered the summation).

  Version 2.40 (1 Sep 1999)
  fixed a bug in the work_gamma procedure - now the gamma values are worked out as specified by the
  equation, tail or no tail. The BCJR algorithm handles tail regions automatically by giving the correct
  boundary conditions for alpha/beta. Actually, the difference in this bugfix will be seen in the use
  with Turbo codes only because the only real problem is with the lack of using APP in the tail region.
  
\warning Static memory requirements: sizeof(real)*(2*(tau+1)*M + tau*M*K + K + N) + sizeof(int)*(2*K+1)*M
\warning Dynamic memory requirements: none
*/
template <class real> class bcjr {
   // internal variables
   int   tau;  //!< Block size in symbols (including any tail bits)
   int   K;    //!< Number of possible input to encoder at any time instant (equals 2^k)
   int   N;    //!< Number of possible outputs of encoder at any time instant (equals 2^n)
   int   M;    //!< Number of possible states of the encoder
   int   nu;   //!< Length of tail in symbols (derived from encoder's memoery order)
   bool  initialised;   //!< Initially false, becomes true after the first call to "decode" when memory is allocated
   bool  startatzero;   //!< True to indicate that the trellis starts in state zero
   bool  endatzero;     //!< True to indicate that the trellis ends in state zero
   // working matrices
   matrix<real>	alpha;   //!< Forward recursion metric: alpha(t,m) = Pr{S(t)=m, Y[1..t]}
   matrix<real>   beta;    //!< Backward recursion metric: beta(t,m) = Pr{Y[t+1..tau] | S(t)=m}
   matrix3<real>	gamma;   //!< Receiver metric: gamma(t-1,m',i) = Pr{S(t)=m(m',i), Y[t] | S(t-1)=m'}
   // temporary arrays/matrices
   matrix<int>    lut_X;   //!< lut_X(m,i) = encoder output if system was in state 'm' and given input 'i'
   matrix<int>    lut_m;   //!< lut_m(m,i) = next state of encoder if system was in state 'm' and given input 'i'
   vector<int>    lut_i;   //!< lut_i(m) = required input to tail off a system in state 'm'
   real	         *rri;    //!< Temporary array for working result - rri(i) = APP of having transmitted (input value) 'i'
   real           *rro;    //!< Temporary array for working result - rro(X) = APP of having transmitted (output value) 'X'
   // memory allocator (for internal use only)
   void allocate();
   // internal functions
   inline real lambda(const int t, const int m);
   inline real sigma(const int t, const int m, const int i);
   // internal procedures
   inline void work_gamma(matrix<double>& R);
   inline void work_gamma(matrix<double>& R, matrix<double>& app);
   inline void work_alpha();
   inline void work_beta();
   inline void work_results(matrix<double>& ri, matrix<double>& ro);
   inline void work_results(matrix<double>& ri);
public:
   bcjr(fsm& encoder, const int tau, const bool startatzero =true, const bool endatzero =true);
   ~bcjr();
   void decode(matrix<double>& R, matrix<double>& ri, matrix<double>& ro);
   void decode(matrix<double>& R, matrix<double>& app, matrix<double>& ri, matrix<double>& ro);
   void fdecode(matrix<double>& R, matrix<double>& ri);
   void fdecode(matrix<double>& R, matrix<double>& app, matrix<double>& ri);
};


// Creation/Destruction routines

/*!
\brief   Creator for class 'bcjr'.
\param	encoder     The finite state machine used to encode the source.
\param   tau         The block length of decoder (including tail bits).
\param   startatzero True if the trellis for the underlying code is known to start at state zero.
\param   endatzero   True if the trellis for the underlying code is known to end at state zero.

  Note that if the trellis is not defined as starting or ending at zero, then it is assumed that
  all starting and ending states (respectively) are equiprobable.
*/
template <class real> inline bcjr<real>::bcjr(fsm& encoder, const int tau, const bool startatzero, const bool endatzero)
   {
   if(tau < 1)
      {
      cerr << "FATAL ERROR (bcjr): MAP decoder block size too small (" << tau << ")\n";
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
      for(int i=0; i<K; i++)
         {
         encoder.reset(mdash);
         int input = i;
         lut_X(mdash, i) = encoder.step(input);
         lut_m(mdash, i) = encoder.state();
         }
      encoder.reset(mdash);
      int i = fsm::tail;
      int X = encoder.step(i);
      lut_X(mdash, i) = X;
      lut_m(mdash, i) = encoder.state();
      lut_i(mdash) = i;
      }
   
   // memory is only allocated in the first decode call, so we have to store the
   // startatzero and endatzero conditions for the intialisation time
   bcjr::startatzero = startatzero;
   bcjr::endatzero = endatzero;
   initialised = false;
   }

/*!
\brief   Destructor for class 'bcjr'.

  Deallocates all temporary storage space reserved by the class. Actually, this routine only needs
  to deallocate 'rri' and 'rro', since matrices are automatically removed when this object is removed.
*/
template <class real> inline bcjr<real>::~bcjr()
   {
   // Deallocate arrays (only if they have been allocated)
   if(initialised)
      {
      delete[] rri;
      delete[] rro;
      }
   }


// Memory allocation
template <class real> void bcjr<real>::allocate()
   {
   // Allocate arrays for working out final metrics
   rri = new real[K];
   rro = new real[N];

   // to save space, gamma is defined from 0 to tau-1, rather than 1 to tau.
   // for this reason, gamma_t (and only gamma_t) is actually written gamma[t-1, ...
   alpha.init(tau+1, M);
   beta.init(tau+1, M);
   gamma.init(tau, M, K);

   // initialise alpha and beta arrays              
   alpha(0, 0) = startatzero ? 1 : 1.0/double(M);
   beta(tau, 0) = endatzero ? 1 : 1.0/double(M);
   for(int m=1; m<M; m++)
      {
      alpha(0, m) = startatzero ? 0 : 1.0/double(M);
      beta(tau, m) = endatzero ? 0 : 1.0/double(M);
      }

   // flag the state of the arrays
   initialised = true;
   }


// Internal functions

//! State probability metric - lambda(t,m) = Pr{S(t)=m, Y[1..tau]}
template <class real> inline real bcjr<real>::lambda(const int t, const int m)
   {
   return alpha(t, m) * beta(t, m);
   }
   
//! Transition probability metric - sigma(t,m,i) = Pr{S(t-1)=m, S(t)=m(m,i), Y[1..tau]}
template <class real> inline real bcjr<real>::sigma(const int t, const int m, const int i)
   {
   int mdash = lut_m(m, i);
   return alpha(t-1, m) * gamma(t-1, m, i) * beta(t, mdash);
   }


// Internal procedures

/*!
\brief   Computes the gamma matrix.
\param   R  R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X

  For all values of t in [1,tau], the gamma values are worked out as specified by the BCJR equation.

\warning The line 'gamma(t-1, mdash, m) = R(t-1, X)' was changed to '+=' on 27 May 1998 to allow for
         the case (as in uncoded Tx) where trellis has parallel paths (more than one path starting at
         the same state and ending at the same state). It was then immediately changed back to '='
         because the BCJR algorithm cannot determine between two parallel paths anyway (algorithm is
         useless in such cases). Same applies to viterbi algorithm.
*/
template <class real> inline void bcjr<real>::work_gamma(matrix<double>& R)
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
\param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X
\param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value) i at time t

  For all values of t in [1,tau], the gamma values are worked out as specified by the BCJR equation.
  This function also makes use of the a priori probabilities associated with the input.
*/
template <class real> inline void bcjr<real>::work_gamma(matrix<double>& R, matrix<double>& app)
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
template <class real> inline void bcjr<real>::work_alpha()
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
      }
   }

/*!
\brief   Computes the beta matrix.

  Beta values only depend on the final values (for t=tau) and on the computed gamma values;
  the matrix is recursively computed. Final beta values are set in the creator and are never
  changed in the object's lifetime.
*/
template <class real> inline void bcjr<real>::work_beta()
   {
   // evaluate all beta values
   for(int t=tau-1; t>=0; t--)
      for(int m=0; m<M; m++)
         {
         beta(t, m) = 0;
         for(int i=0; i<K; i++)
            {
            int mdash = lut_m(m, i); 
            beta(t, m) += beta(t+1, mdash) * gamma(t, m, i);
            }
         }
   }

/*!
\brief   Computes the final results for the BCJR algorithm.
\param   ri    ri(t-1, i) is the probability that we transmitted (input value) i at time t
\param   ro    ro(t-1, X) is the probability that we transmitted (output value) X at time t

  Once we have worked out the gamma, alpha, and beta matrices, we are in a position to compute
  Py (the probability of having received the received sequence of modulation symbols). Next, we
  compute the results by doing the appropriate summations on sigma.

\warning Initially, I used to work out the delta probability as 'delta = lambda(t-1, mdash)/Py * sigma(t, mdash, m)/Py'.
         I suspected this reasoning to be false, and am now working the delta value as 'delta = sigma(t, mdash, m)/Py'.
         This makes sense because the sigma values already take into account the probability of being in state mdash
         before the transition being considered (we care about the transition because this determines the input and
         output symbols represented).
*/
template <class real> inline void bcjr<real>::work_results(matrix<double>& ri, matrix<double>& ro)
   {
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // Work out final results
   for(int t=1; t<=tau; t++)
      {
      // initialise results (we compute them first as 'real' values)
      for(int i=0; i<K; i++)
         rri[i] = 0;
      for(int X=0; X<N; X++)
         rro[X] = 0;
      // work out results
      for(int mdash=0; mdash<M; mdash++)	// for each possible state at time t-1
         for(int i=0; i<K; i++)	// for each possible input, given the state we were in
            {
            int X = lut_X(mdash, i);
            real delta = sigma(t, mdash, i)/Py;
            rri[i] += delta;
            rro[X] += delta;
            }
      // copy results into their final place (convert from 'real' to 'double')
      for(int i=0; i<K; i++)
         ri(t-1, i) = rri[i];
      for(int X=0; X<N; X++)
         ro(t-1, X) = rro[X];
      }
   }

/*!
\brief   Computes the final results for the BCJR algorithm (input statistics only).
\param   ri    ri(t-1, i) is the probability that we transmitted (input value) i at time t

  Once we have worked out the gamma, alpha, and beta matrices, we are in a position to compute
  Py (the probability of having received the received sequence of modulation symbols). Next, we
  compute the results by doing the appropriate summations on sigma.
*/
template <class real> inline void bcjr<real>::work_results(matrix<double>& ri)
   {
   // Compute probability of received sequence
   real Py = 0;
   for(int mdash=0; mdash<M; mdash++) // for each possible ending state
      Py += lambda(tau, mdash);
   // Work out final results
   for(int t=1; t<=tau; t++)
      for(int i=0; i<K; i++)	// for each possible input, given the state we were in
         {
         // initialise results (we compute them first as 'real' values)
         real delta = 0;
         for(int mdash=0; mdash<M; mdash++)	// for each possible state at time t-1
            delta += sigma(t, mdash, i);
         // copy results into their final place (convert from 'real' to 'double')
         ri(t-1, i) = delta/Py;
         }
   }

// User procedures

/*!
\brief   Wrapping function for decoding a block.
\param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X
\param   ri 	ri(t-1, i) is the a posteriori probability of having transmitted (input value) i at time t (result)
\param   ro 	ro(t-1, X) = (result) a posteriori probability of having transmitted (output value) X at time t (result)
*/
template <class real> inline void bcjr<real>::decode(matrix<double>& R, matrix<double>& ri, matrix<double>& ro)
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
\param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X
\param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value) i at time t
\param   ri 	ri(t-1, i) is the a posteriori probability of having transmitted (input value) i at time t (result)
\param   ro 	ro(t-1, X) = (result) a posteriori probability of having transmitted (output value) X at time t (result)
*/
template <class real> inline void bcjr<real>::decode(matrix<double>& R, matrix<double>& app, matrix<double>& ri, matrix<double>& ro)
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
\param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X
\param   ri 	ri(t-1, i) is the a posteriori probability of having transmitted (input value) i at time t (result)
*/
template <class real> inline void bcjr<real>::fdecode(matrix<double>& R, matrix<double>& ri)
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
\param   R     R(t-1, X) is the probability of receiving "whatever we received" at time t, having transmitted X
\param   app   app(t-1, i) is the 'a priori' probability of having transmitted (input value) i at time t
\param   ri 	ri(t-1, i) is the a posteriori probability of having transmitted (input value) i at time t (result)
*/
template <class real> inline void bcjr<real>::fdecode(matrix<double>& R, matrix<double>& app, matrix<double>& ri)
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

#endif

