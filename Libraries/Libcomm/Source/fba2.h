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
   /*! \name User-defined parameters */
   int   N;    //!< The transmitted block size in symbols
   int   n;    //!< The number of bits encoding each q-ary symbol
   int   q;    //!< The number of symbols in the q-ary alphabet
   int   I;    //!< The maximum number of insertions considered before every transmission
   int   xmax; //!< The maximum allowed drift is \f$ \pm x_{max} \f$
   // @}
   /*! \name Internally-used objects */
   int   dxmin;         //!< Offset for deltax index in gamma matrix
   int   dxmax;         //!< Maximum value for deltax index in gamma matrix
   bool  initialised;   //!< Flag to indicate when memory is allocated
   libbase::matrix<real>   m_alpha;    //!< Forward recursion metric
   libbase::matrix<real>   m_beta;     //!< Backward recursion metric
   mutable libbase::matrix< libbase::matrix<real> >  m_gamma;    //!< Receiver metric
   mutable libbase::matrix3<bool>  m_cached;    //!< Flag for caching of receiver metric
   // @}
private:
   /*! \name Internal functions */
   real compute_gamma(int d, int i, int x, int deltax, const libbase::vector<sig>& r) const;
   // index-shifting access internal use
   real& alpha(int i, int x) { return m_alpha(i,x+xmax); };
   real& beta(int i, int x) { return m_beta(i,x+xmax); };
   real& gamma(int d, int i, int x, int deltax) { return m_gamma(d,i)(x+xmax,deltax-dxmin); };
   // const versions of above
   real alpha(int i, int x) const { return m_alpha(i,x+xmax); };
   real beta(int i, int x) const { return m_beta(i,x+xmax); };
   real gamma(int d, int i, int x, int deltax) const { return m_gamma(d,i)(x+xmax,deltax-dxmin); };
   // memory allocation
   void allocate();
   // @}
protected:
   /*! \name Internal functions */
   // getters for parameters
   int get_I() const { return I; };
   int get_xmax() const { return xmax; };
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
   fba2(int N, int n, int q, int I, int xmax) { init(N, n, q, I, xmax); };
   virtual ~fba2() {};
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax);

   // decode functions
   void prepare(const libbase::vector<sig>& r);
   void work_results(const libbase::vector<sig>& r, libbase::matrix<real>& ptable) const;
};

template <class real, class sig> real fba2<real,sig>::compute_gamma(int d, int i, int x, int deltax, const libbase::vector<sig>& r) const
   {
   if(!m_cached(i,x+xmax,deltax-dxmin))
      {
      m_cached(i,x+xmax,deltax-dxmin) = true;
      for(int d=0; d<q; d++)
         m_gamma(d,i)(x+xmax,deltax-dxmin) = Q(d,i,r.extract(n*i+x,n+deltax));
      }
   return m_gamma(d,i)(x+xmax,deltax-dxmin);
   }

}; // end namespace

#endif
