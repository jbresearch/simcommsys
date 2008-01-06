#ifndef __bcjr_h
#define __bcjr_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"

#include "sigspace.h"
#include "fsm.h"

#include <math.h>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
   \brief   Bahl-Cocke-Jelinek-Raviv (BCJR) decoding algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   All internal metrics are held as type 'real', which is user-defined. This allows internal working
   at any required level of accuracy. This is required because the internal matrics have a very wide
   dynamic range, which increases exponentially with block size 'tau'. Actually, the required range
   is within [1,0), but very large exponents are required.

   \warning Static memory requirements: sizeof(real)*(2*(tau+1)*M + tau*M*K + K + N) + sizeof(int)*(2*K+1)*M
   \warning Dynamic memory requirements: none

   \version 2.00 (26 Feb 1999)
   updated by redefining the gamma matrix to contain only the connected set of mdash,m (reducing memory
   by a factor of M/K) and by computing alpha and beta accordingly (a speedup of M/K).

   \version 2.10 (4 Mar 1999)
   memory is only allocated in the first call to "decode". This is more efficient for the MPI strategy
   with a root server which only collects results.

   \version 2.20 (5 Mar 1999)
   modified the definition of nu (defined it as 0 for endatzero==false), in order to work out the gamma
   matrix correctly. Note that all previous results are invalid for endatzero==false!

   \version 2.30 (8 Mar 1999)
   added a faster decode routine (fdecode) which does not produce a posteriori statistics on the
   decoder's output.

   \version 2.31 (8 Mar 1999)
   optimised fdecode (reordered the summation).

   \version 2.40 (1 Sep 1999)
   fixed a bug in the work_gamma procedure - now the gamma values are worked out as specified by the
   equation, tail or no tail. The BCJR algorithm handles tail regions automatically by giving the correct
   boundary conditions for alpha/beta. Actually, the difference in this bugfix will be seen in the use
   with Turbo codes only because the only real problem is with the lack of using APP in the tail region.

   \version 2.41 (24 Oct 2001)
   moved most functions to the cpp file rather than the header, in order to be able to
   compile the header with Microsoft Extensions. Naturally compilation is faster, but this
   also requires realizations of the class within the cpp file. This was done for mpreal,
   mpgnu and logreal.

   \version 2.42 (23 Feb 2002)
   made some minor changes to work_results(ri) to speed up things a little.

   \version 2.43 (1 Mar 2002)
   edited the classes to be compileable with Microsoft extensions enabled - in practice,
   the major change is in for() loops, where MS defines scope differently from ANSI.
   Rather than taking the loop variables into function scope, we chose to wrap around the
   offending for() loops.

   \version 2.44 (6 Mar 2002)
   changed use of iostream from global to std namespace.

   \version 2.45 (13 Mar 2002)
   added protected init() function and default constructor, which allow derived classes
   to be re-used (needed for serialization).

   \version 2.46 (4 Apr 2002)
   removed 'inline' keyword from internal functions and procedures. Also made version variable
   a private static const member.

   \version 2.47 (18 Apr 2005)
   removed internal temporary arrays 'rri' and 'rro'. Since the matrix and vector classes
   have been optimised (and only do range checking in the Debug build), there is no real
   need to use arrays any more. This makes the code cleaner and less error-prone.
   Also cleaned up the 'work_results(ri,ro)' function, so that the results are accumulated
   into a double value; this mirrirs the way that 'work_results(ri)' works.

   \version 2.50 (18 Apr 2005)
   added a second template class 'dbl', which defaults to 'double', to allow other
   numerical representations for externally-transferred statistics. This became necessary
   for the parallel decoding structure, where the range of extrinsic information is much
   larger than for serial decoding; furthermore, this range increases with the number of
   iterations performed.

   \version 2.51 (21 Jul 2006)
   added support for circular decoding - when working alpha and beta vectors, the set of
   probabilities at the end state are normalized and used to replace the corresponding
   set for the initial state. To support this, a new flag 'circular' has been added to
   the constructor, defaulting to false. In addition, a new function reset() has been
   added, to provide a mechanism for the turbo codec to reset the start- and end-state
   probabilities between frames.

   \version 2.52 (1 Aug 2006)
   added internal normalization of alpha and beta metrics, as in Matt Valenti's CML
   Theory slides; this is an attempt at solving the numerical range problems currently
   experienced in multiple (sets>2) Turbo codes.

   \version 2.53 (2 Aug 2006)
   - modified internal normalization - rather than dividing by the value for the first
   symbol, we now determine the maximum value over all symbols and divide by that. This
   should avoid problems when the metric for the first symbol is very small.
   - modified work_results functions, so that internal results are worked out as type
   'dbl' rather than 'double'. It is clear that this was meant to be so all along.
   - added normalization function for use by derived classes (such as turbo);
   rather than normalizing the a-priori and a-posteriori probabilities here, this
   is left for derived classes to do - the reason behind this is that this class
   should not be responsible for its inputs, but whoever is providing them is.

   \version 2.54 (3 Aug 2006)
   modified functions 'fdecode' & 'decode' (and consequently a number of internal
   functions; notably 'work_gamma') to indicate within the prototype which parameters
   are input (by making them const). While this should not change any results, it is a
   forward step to simplify debugging, This was necessitated by turbo 2.47.

   \version 2.60 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 2.53 (6 Jan 2008)
   - removed various redundant blocks, a remnant from old VS
   - modified decoding behaviour with circular trellises: instead of replacing the start-
     state probabilities with the end-state probabilities at the end of the forward pass
     (and similarly replacing the end-state with the start-state probabilities at the
     end of the backward pass), we now do the exchange before we start the turn.
     Consequently, this requires a slightly different metric initialization.
*/

template <class real, class dbl=double> class bcjr {
   // internal variables
   int   tau;  //!< Block size in symbols (including any tail bits)
   int   K;    //!< Number of possible input to encoder at any time instant (equals 2^k)
   int   N;    //!< Number of possible outputs of encoder at any time instant (equals 2^n)
   int   M;    //!< Number of possible states of the encoder
   int   nu;   //!< Length of tail in symbols (derived from encoder's memoery order)
   bool  initialised;   //!< Initially false, becomes true after the first call to "decode" when memory is allocated
   bool  startatzero;   //!< True to indicate that the trellis starts in state zero
   bool  endatzero;     //!< True to indicate that the trellis ends in state zero
   bool  circular;      //!< True to indicate circular trellis decoding
   // working matrices
   libbase::matrix<real>   alpha;   //!< Forward recursion metric: alpha(t,m) = Pr{S(t)=m, Y(1..t)}
   libbase::matrix<real>   beta;    //!< Backward recursion metric: beta(t,m) = Pr{Y(t+1..tau) | S(t)=m}
   libbase::matrix3<real>  gamma;   //!< Receiver metric: gamma(t-1,m',i) = Pr{S(t)=m(m',i), Y(t) | S(t-1)=m'}
   // temporary arrays/matrices
   libbase::matrix<int>    lut_X;   //!< lut_X(m,i) = encoder output if system was in state 'm' and given input 'i'
   libbase::matrix<int>    lut_m;   //!< lut_m(m,i) = next state of encoder if system was in state 'm' and given input 'i'
   libbase::vector<int>    lut_i;   //!< lut_i(m) = required input to tail off a system in state 'm'
   // memory allocator (for internal use only)
   void allocate();
   // internal functions
   real lambda(const int t, const int m);
   real sigma(const int t, const int m, const int i);
   // internal procedures
   void work_gamma(const libbase::matrix<dbl>& R);
   void work_gamma(const libbase::matrix<dbl>& R, const libbase::matrix<dbl>& app);
   void work_alpha();
   void work_beta();
   void work_results(libbase::matrix<dbl>& ri, libbase::matrix<dbl>& ro);
   void work_results(libbase::matrix<dbl>& ri);
protected:
   // normalization function for derived classes
   static void normalize(libbase::matrix<dbl>& r);
   // main initialization routine - constructor essentially just calls this
   void init(fsm& encoder, const int tau, const bool startatzero, const bool endatzero, const bool circular =false);
   // reset start- and end-state probabilities
   void reset();
   bcjr();
public:
   // constructor & destructor
   bcjr(fsm& encoder, const int tau, const bool startatzero =true, const bool endatzero =true, const bool circular =false);
   ~bcjr() {};
   // decode functions
   void decode(const libbase::matrix<dbl>& R, libbase::matrix<dbl>& ri, libbase::matrix<dbl>& ro);
   void decode(const libbase::matrix<dbl>& R, const libbase::matrix<dbl>& app, libbase::matrix<dbl>& ri, libbase::matrix<dbl>& ro);
   void fdecode(const libbase::matrix<dbl>& R, libbase::matrix<dbl>& ri);
   void fdecode(const libbase::matrix<dbl>& R, const libbase::matrix<dbl>& app, libbase::matrix<dbl>& ri);
};

}; // end namespace

#endif

