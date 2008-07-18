#ifndef __fba2_h
#define __fba2_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "multi_array.h"
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

template <class real, class sig, bool normalize>
class fba2 {
private:
   /*! \name Internally-used types */
   typedef libbase::vector<sig>     array1s_t;
   typedef libbase::matrix<real>    array2r_old_t;
   typedef boost::assignable_multi_array<real,2> array2r_t;
   typedef boost::assignable_multi_array<real,4> array4r_t;
   typedef boost::assignable_multi_array<bool,3> array3b_t;
   // @}
private:
   /*! \name User-defined parameters */
   int   N;       //!< The transmitted block size in symbols
   int   n;       //!< The number of bits encoding each q-ary symbol
   int   q;       //!< The number of symbols in the q-ary alphabet
   int   I;       //!< The maximum number of insertions considered before every transmission
   int   xmax;    //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int   dxmax;   //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   double th_inner;  //!< Threshold factor for inner cycle
   double th_outer;  //!< Threshold factor for outer cycle
   // @}
   /*! \name Internally-used objects */
   int   dmin;          //!< Offset for deltax index in gamma matrix
   int   dmax;          //!< Maximum value for deltax index in gamma matrix
   bool  initialised;   //!< Flag to indicate when memory is allocated
   bool  cache_enabled; //!< Flag to indicate when cache is usable
   array2r_t alpha;     //!< Forward recursion metric
   array2r_t beta;      //!< Backward recursion metric
   mutable array4r_t gamma;   //!< Receiver metric
   mutable array3b_t cached;  //!< Flag for caching of receiver metric
#ifndef NDEBUG
   mutable int gamma_calls;   //!< Number of gamma computations
   mutable int gamma_misses;  //!< Number of gamma computations causing a cache miss
#endif
   // @}
private:
   /*! \name Internal functions */
   real compute_gamma(int d, int i, int x, int deltax, const array1s_t& r) const;
   // memory allocation
   void allocate();
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real R(int d, int i, const array1s_t& r) const = 0;
   // decode functions
   void work_gamma(const array1s_t& r);
   void work_alpha(const array1s_t& r);
   void work_beta(const array1s_t& r);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2() { initialised = false; };
   virtual ~fba2() {};
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner, double th_outer);

   // decode functions
   void prepare(const array1s_t& r);
   void work_results(const array1s_t& r, array2r_old_t& ptable) const;
};

template <class real, class sig, bool normalize>
real fba2<real,sig,normalize>::compute_gamma(int d, int i, int x, int deltax, const array1s_t& r) const
   {
   if(!cache_enabled)
      return R(d,i,r.extract(n*i+x,n+deltax));

   if(!cached[i][x][deltax])
      {
      cached[i][x][deltax] = true;
      for(int d=0; d<q; d++)
         gamma[d][i][x][deltax] = R(d,i,r.extract(n*i+x,n+deltax));
#ifndef NDEBUG
      gamma_misses++;
#endif
      }
#ifndef NDEBUG
   gamma_calls++;
#endif

   return gamma[d][i][x][deltax];
   }

}; // end namespace

#endif
