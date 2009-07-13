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

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 Implements the forward-backward algorithm for a HMM, as required for the
 new decoder for Davey & McKay's inner codes, originally introduced in
 "Watermark Codes: Reliable communication over Insertion/Deletion channels",
 Trans. IT, 47(2), Feb 2001.
 */

template <class real, class sig, bool norm>
class fba2 {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   typedef boost::assignable_multi_array<real, 2> array2r_t;
   typedef boost::assignable_multi_array<real, 4> array4r_t;
   typedef boost::assignable_multi_array<bool, 3> array3b_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef fba2<real, sig, norm> This;
private:
   /*! \name User-defined parameters */
   int N; //!< The transmitted block size in symbols
   int n; //!< The number of bits encoding each q-ary symbol
   int q; //!< The number of symbols in the q-ary alphabet
   int I; //!< The maximum number of insertions considered before every transmission
   int xmax; //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int dxmax; //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   double th_inner; //!< Threshold factor for inner cycle
   double th_outer; //!< Threshold factor for outer cycle
   // @}
   /*! \name Internally-used objects */
   int dmin; //!< Offset for deltax index in gamma matrix
   int dmax; //!< Maximum value for deltax index in gamma matrix
   bool initialised; //!< Flag to indicate when memory is allocated
   bool cache_enabled; //!< Flag to indicate when cache is usable
   array2r_t alpha; //!< Forward recursion metric
   array2r_t beta; //!< Backward recursion metric
   mutable array4r_t gamma; //!< Receiver metric
   mutable array3b_t cached; //!< Flag for caching of receiver metric
   array1s_t r; //!< Copy of received sequence, for lazy computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for lazy computation of gamma
#ifndef NDEBUG
   mutable int gamma_calls; //!< Number of gamma computations
   mutable int gamma_misses; //!< Number of gamma computations causing a cache miss
#endif
   // @}
private:
   /*! \name Internal functions */
   real compute_gamma(int d, int i, int x, int deltax) const;
   real get_gamma(int d, int i, int x, int deltax) const;
   // memory allocation
   void allocate();
   void free();
   void reset_cache() const;
   // @}
protected:
   /*! \name Internal functions */
   // handles for channel-specific metrics - to be implemented by derived classes
   virtual real R(int d, int i, const array1s_t& r) const = 0;
   // decode functions
   void work_gamma(const array1s_t& r, const array1vd_t& app);
   void work_gamma(const array1s_t& r);
   void work_alpha(int rho);
   void work_beta(int rho);
   void work_results(int rho, array1vr_t& ptable) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2()
      {
      initialised = false;
      }
   virtual ~fba2()
      {
      }
   // @}

   // main initialization routine - constructor essentially just calls this
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer);

   // decode functions
   void decode(const array1s_t& r, const array1vd_t& app, array1vr_t& ptable);
   void decode(const array1s_t& r, array1vr_t& ptable);
};

template <class real, class sig, bool norm>
inline real fba2<real, sig, norm>::compute_gamma(int d, int i, int x,
      int deltax) const
   {
   real result = R(d, i, r.extract(n * i + x, n + deltax));
   if (app.size() > 0)
      result *= app(i)(d);
   return result;
   }

template <class real, class sig, bool norm>
real fba2<real, sig, norm>::get_gamma(int d, int i, int x, int deltax) const
   {
   if (!cache_enabled)
      return compute_gamma(d, i, x, deltax);

   if (!cached[i][x][deltax])
      {
      cached[i][x][deltax] = true;
      for (int d = 0; d < q; d++)
         gamma[d][i][x][deltax] = compute_gamma(d, i, x, deltax);
#ifndef NDEBUG
      gamma_misses++;
#endif
      }
#ifndef NDEBUG
   gamma_calls++;
#endif

   return gamma[d][i][x][deltax];
   }

} // end namespace

#endif
