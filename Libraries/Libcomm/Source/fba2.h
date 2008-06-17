#ifndef __fba2_h
#define __fba2_h

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
   \brief   Alternative Forward-Backward Algorithm.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements the forward-backward algorithm for a HMM, as required for the
   new decoder for Davey & McKay's inner codes, originally introduced in
   "Watermark Codes: Reliable communication over Insertion/Deletion channels",
   Trans. IT, 47(2), Feb 2001.
*/

template <class real, class sig=sigspace> class fba2 {
private:
   /*! \name Static parameters */
   static const double  threshold_inner;
   static const double  threshold_outer;
   // @}
   /*! \name User-defined parameters */
   int   N;       //!< The transmitted block size in symbols
   int   n;       //!< The number of bits encoding each q-ary symbol
   int   q;       //!< The number of symbols in the q-ary alphabet
   int   I;       //!< The maximum number of insertions considered before every transmission
   int   xmax;    //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int   dxmax;   //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   // @}
   /*! \name Internally-used objects */
   int   dmin;          //!< Offset for deltax index in gamma matrix
   int   dmax;          //!< Maximum value for deltax index in gamma matrix
   bool  initialised;   //!< Flag to indicate when memory is allocated
   bool  cache_enabled; //!< Flag to indicate when cache is usable
   libbase::matrix<real>   m_alpha;    //!< Forward recursion metric
   libbase::matrix<real>   m_beta;     //!< Backward recursion metric
   mutable libbase::matrix< libbase::matrix<real> >  m_gamma;    //!< Receiver metric
   mutable libbase::matrix3<bool>  m_cached;    //!< Flag for caching of receiver metric
#ifndef NDEBUG
   mutable int gamma_calls;   //!< Number of gamma computations
   mutable int gamma_misses;  //!< Number of gamma computations causing a cache miss
#endif
   // @}
private:
   /*! \name Internal functions */
   real compute_gamma(int d, int i, int x, int deltax, const libbase::vector<sig>& r) const;
   // index-shifting access internal use
   real& alpha(int i, int x) { return m_alpha(i,x+xmax); };
   real& beta(int i, int x) { return m_beta(i,x+xmax); };
   //real& gamma(int d, int i, int x, int deltax) { return m_gamma(d,i)(x+xmax,deltax-dmin); };
   // const versions of above
   real alpha(int i, int x) const { return m_alpha(i,x+xmax); };
   real beta(int i, int x) const { return m_beta(i,x+xmax); };
   //real gamma(int d, int i, int x, int deltax) const { return m_gamma(d,i)(x+xmax,deltax-dmin); };
   // memory allocation
   void allocate();
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real Q(int d, int i, const libbase::vector<sig>& r) const = 0;
   // decode functions
   void work_gamma(const libbase::vector<sig>& r);
   void work_alpha(const libbase::vector<sig>& r);
   void work_beta(const libbase::vector<sig>& r);
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2() { initialised = false; };
   // @}
public:
   /*! \name Constructors / Destructors */
   fba2(int N, int n, int q, int I, int xmax, int dxmax) { init(N, n, q, I, xmax, dxmax); };
   virtual ~fba2() {};
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax, int dxmax);

   // decode functions
   void prepare(const libbase::vector<sig>& r);
   void work_results(const libbase::vector<sig>& r, libbase::matrix<real>& ptable) const;
};

template <class real, class sig> real fba2<real,sig>::compute_gamma(int d, int i, int x, int deltax, const libbase::vector<sig>& r) const
   {
   if(!cache_enabled)
      return Q(d,i,r.extract(n*i+x,n+deltax));

   if(!m_cached(i,x+xmax,deltax-dmin))
      {
      m_cached(i,x+xmax,deltax-dmin) = true;
      for(int d=0; d<q; d++)
         m_gamma(d,i)(x+xmax,deltax-dmin) = Q(d,i,r.extract(n*i+x,n+deltax));
#ifndef NDEBUG
      gamma_misses++;
#endif
      }
#ifndef NDEBUG
   gamma_calls++;
#endif

   return m_gamma(d,i)(x+xmax,deltax-dmin);
   }

}; // end namespace

#endif
