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

   All internal metrics are held as type 'real', which is user-defined. This
   allows internal working at any required level of accuracy. This is required
   because the internal matrics have a very wide dynamic range, which increases
   exponentially with block size 'tau'. Actually, the required range is within
   [1,0), but very large exponents are required.

   The second template class 'dbl', which defaults to 'double', allows other
   numerical representations for externally-transferred statistics. This became
   necessary for the parallel decoding structure, where the range of extrinsic
   information is much larger than for serial decoding; furthermore, this range
   increases with the number of iterations performed.

   \warning
      - Static memory requirements:
         sizeof(real)*(2*(tau+1)*M + tau*M*K + K + N) + sizeof(int)*(2*K+1)*M
      - Dynamic memory requirements:
         none

   \note Memory is only allocated in the first call to "decode". This is more
         efficient for the parallel simulator strategy with a master which only
         collects results.

   \version 2.00 (26 Feb 1999)
   updated by redefining the gamma matrix to contain only the connected set of mdash,m (reducing memory
   by a factor of M/K) and by computing alpha and beta accordingly (a speedup of M/K).

   \version 2.40 (1 Sep 1999)
   fixed a bug in the work_gamma procedure - now the gamma values are worked out as specified by the
   equation, tail or no tail. The BCJR algorithm handles tail regions automatically by giving the correct
   boundary conditions for alpha/beta. Actually, the difference in this bugfix will be seen in the use
   with Turbo codes only because the only real problem is with the lack of using APP in the tail region.

   \version 2.52 (1 Aug 2006)
   added internal normalization of alpha and beta metrics, as in Matt Valenti's CML
   Theory slides; this is an attempt at solving the numerical range problems currently
   experienced in multiple (sets>2) Turbo codes.

   \version 2.53 (2 Aug 2006)
   - modified internal normalization - rather than dividing by the value for the first
   symbol, we now determine the maximum value over all symbols and divide by that. This
   should avoid problems when the metric for the first symbol is very small.
   - added normalization function for use by derived classes (such as turbo);
   rather than normalizing the a-priori and a-posteriori probabilities here, this
   is left for derived classes to do - the reason behind this is that this class
   should not be responsible for its inputs, but whoever is providing them is.

   \version 2.51 (21 Jul 2006)
   added support for circular decoding - when working alpha and beta vectors, the set of
   probabilities at the end state are normalized and used to replace the corresponding
   set for the initial state. To support this, a new flag 'circular' has been added to
   the constructor, defaulting to false. In addition, a new function reset() has been
   added, to provide a mechanism for the turbo codec to reset the start- and end-state
   probabilities between frames.

   \version 2.60 (6 Jan 2008)
   - modified decoding behaviour with circular trellises: instead of replacing the start-
     state probabilities with the end-state probabilities at the end of the forward pass
     (and similarly replacing the end-state with the start-state probabilities at the
     end of the backward pass), we now do the exchange before we start the turn.
     Consequently, this requires a slightly different metric initialization.
   - observed bug in handling of circular decoding: since the same object is used over
     different sets of the turbo structure, the probabilities copied over between end-
     state and start-state were actually getting mixed up between iterations. Fixing this
     as follows:
      - removed flags startatzero, endatzero, and circular; these are actually properties
        of the turbo code, rather than the bcjr object itself, and are only relevant for
        two things: (i) initializing the metric endpoints and (ii) determining the length
        of the tail sequence
      - modified reset() and created similar new protected functions to cater for (i),
        getting and setting the end-state probabilities, to be used by the turbo system
      - removed the 'nu' variable, which held the length of the tail sequence, and the
        'lut_i' table, since these are not really required.
*/

template <class real, class dbl=double>
class bcjr {
private:
   /*! \name Internally-used types */
   typedef libbase::matrix<int>     array2i_t;
   typedef libbase::vector<dbl>     array1d_t;
   typedef libbase::matrix<dbl>     array2d_t;
   typedef libbase::matrix<real>    array2r_t;
   typedef libbase::matrix3<real>   array3r_t;
   // @}
   /*! \name Internal variables */
   int   tau;           //!< Input block size in symbols (including tail)
   int   K;             //!< Input alphabet size
   int   N;             //!< Output alphabet size
   int   M;             //!< Number of encoder states
   bool  initialised;   //!< Flag to indicate when memory is allocated
   // @}
   /*! \name Working matrices */
   //! Forward recursion metric: alpha(t,m) = Pr{S(t)=m, Y(1..t)}
   array2r_t   alpha;
   //! Backward recursion metric: beta(t,m) = Pr{Y(t+1..tau) | S(t)=m}
   array2r_t   beta;
   //! Receiver metric: gamma(t-1,m',i) = Pr{S(t)=m(m',i), Y(t) | S(t-1)=m'}
   array3r_t   gamma;
   // @}
   /*! \name Temporary (cache) matrices */
   //! lut_X(m,i) = encoder output, given state 'm' and input 'i'
   array2i_t   lut_X;
   //! lut_m(m,i) = next state, given state 'm' and input 'i'
   array2i_t   lut_m;
   // @}
private:
   // memory allocator (for internal use only)
   void allocate();
   // internal functions
   real lambda(const int t, const int m);
   real sigma(const int t, const int m, const int i);
   // internal procedures
   void work_gamma(const array2d_t& R);
   void work_gamma(const array2d_t& R, const array2d_t& app);
   void work_alpha();
   void work_beta();
   void work_results(array2d_t& ri, array2d_t& ro);
   void work_results(array2d_t& ri);
protected:
   // normalization function for derived classes
   static void normalize(array2d_t& r);
   // main initialization routine - constructor essentially just calls this
   void init(fsm& encoder, const int tau);
   // get start- and end-state probabilities
   array1d_t getstart() const;
   array1d_t getend() const;
   // set start- and end-state probabilities - equiprobable
   void setstart();
   void setend();
   // set start- and end-state probabilities - known state
   void setstart(int state);
   void setend(int state);
   // set start- and end-state probabilities - direct
   void setstart(const array1d_t& p);
   void setend(const array1d_t& p);
   // default constructor
   bcjr() { initialised = false; };
public:
   // constructor & destructor
   bcjr(fsm& encoder, const int tau) { init(encoder, tau); };
   ~bcjr() {};
   // decode functions
   void decode(const array2d_t& R, array2d_t& ri, array2d_t& ro);
   void decode(const array2d_t& R, const array2d_t& app, array2d_t& ri, array2d_t& ro);
   void fdecode(const array2d_t& R, array2d_t& ri);
   void fdecode(const array2d_t& R, const array2d_t& app, array2d_t& ri);
};

}; // end namespace

#endif

