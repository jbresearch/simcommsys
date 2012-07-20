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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __fba2_cuda_h
#define __fba2_cuda_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "fsm.h"
#include "cuda-all.h"
#include "instrumented.h"
#include "channel/bsid.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show index computation for gamma vector
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Symbol-Level Forward-Backward Algorithm for CUDA.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * new decoder for Davey & McKay's inner codes, originally introduced in
 * "Watermark Codes: Reliable communication over Insertion/Deletion channels",
 * Trans. IT, 47(2), Feb 2001.
 *
 * \warning Do not use shorthand for class hierarchy, as these are not
 * interpreted properly by NVCC.
 */

template <class receiver_t, class sig, class real>
class fba2 {
public:
   // forward definition
   class metric_computer;
public:
   /*! \name Type definitions */
   // Device-based types - data containers
   typedef cuda::vector<sig> dev_array1s_t;
   typedef cuda::vector<real> dev_array1r_t;
   typedef cuda::matrix<real> dev_array2r_t;
   typedef cuda::matrix<bool> dev_array2b_t;
   typedef cuda::value<fba2<receiver_t, sig, real>::metric_computer>
         dev_object_t;
   // Device-based types - references
   typedef cuda::vector_reference<sig> dev_array1s_ref_t;
   typedef cuda::vector_reference<real> dev_array1r_ref_t;
   typedef cuda::matrix_reference<real> dev_array2r_ref_t;
   typedef cuda::matrix_reference<bool> dev_array2b_ref_t;
   typedef cuda::value_reference<fba2<receiver_t, sig, real>::metric_computer>
         dev_object_ref_t;
   // Host-based types
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   // @}
public:
   /*! \name Metric computation */
   /*! \brief Class encapsulating metric computation to be done on device.
    * Note this class is publicly defined, so that it is known by (global)
    * CUDA kernels.
    */
   class metric_computer {
   public:
      /*! \name Internally-used objects */
      mutable receiver_t receiver; //!< Inner code receiver metric computation
      dev_array2r_ref_t alpha; //!< Forward recursion metric
      dev_array2r_ref_t beta; //!< Backward recursion metric
      mutable dev_array1r_ref_t gamma; //!< Receiver metric
      mutable dev_array2b_ref_t cached; //!< Flag for globalstore of receiver metric
      dev_array1s_ref_t r; //!< Copy of received sequence, for lazy computation of gamma
      dev_array2r_ref_t app; //!< Copy of a-priori statistics, for lazy computation of gamma
      int dmin; //!< Offset for deltax index in gamma matrix
      int dmax; //!< Maximum value for deltax index in gamma matrix
      // @}
      /*! \name User-defined parameters */
      real th_inner; //!< Threshold factor for inner cycle
      real th_outer; //!< Threshold factor for outer cycle
      int N; //!< The transmitted block size in symbols
      int n; //!< The number of bits encoding each q-ary symbol
      int q; //!< The number of symbols in the q-ary alphabet
      int I; //!< The maximum number of insertions considered before every transmission
      int xmax; //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
      int dxmax; //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
      struct {
         bool norm; //!< Flag to indicate if metrics should be normalized between time-steps
         bool batch; //!< Flag indicating use of batch receiver interface
         bool lazy; //!< Flag indicating lazy computation of gamma metric
         bool globalstore; //!< Flag indicating we will try to cache lazily computed gamma values
      } flags;
      // @}
   public:
      /*! \name Internal functions - computer */
      // Methods for device and host
#ifdef __CUDACC__
      __device__ __host__
#endif
      int get_gamma_index(int d, int i, int x, int deltax) const
         {
         // gamma has indices (d,i,x,deltax) where:
         //    d in [0, q-1], i in [0, N-1], x in [-xmax, xmax], and
         //    deltax in [dmin, dmax] = [max(-n,-dxmax), min(nI,dxmax)]
         const int pitch3 = (dmax - dmin + 1);
         const int pitch2 = pitch3 * (2 * xmax + 1);
         const int pitch1 = pitch2 * N;
         cuda_assert(d >= 0 && d < q);
         cuda_assert(i >= 0 && i < N);
         cuda_assert(x >= -xmax && x <= xmax);
         cuda_assert(deltax >= dmin && deltax <= dmax);
         const int off1 = d;
         const int off2 = i;
         const int off3 = x + xmax;
         const int off4 = deltax - dmin;
         const int ndx = off1 * pitch1 + off2 * pitch2 + off3 * pitch3 + off4;
#ifndef __CUDA_ARCH__
         // host code path only
#if DEBUG>=2
         std::cerr << "(" << d << "," << i << "," << x << "," << deltax << ":"
         << ndx << ")";
#endif
#endif
         return ndx;
         }
      // Device-only methods
#ifdef __CUDACC__
      //! Compute gamma metric using independent receiver interface
      __device__
      real compute_gamma_single(int d, int i, int x, int deltax, const dev_array1s_ref_t& r, const dev_array2r_ref_t& app) const
         {
         // determine received segment to extract
         const int start = xmax + n * i + x;
         const int length = n + deltax;
         // call receiver method
         real result = receiver.R(d, i, r.extract(start, length));
         // apply priors if applicable
         if (app.size() > 0)
            {
            result *= real(app(i,d));
            }
         return result;
         }
      //! Compute gamma metric using batch receiver interface
      __device__
      void compute_gamma_batch(int d, int i, int x, vector_reference<libcomm::bsid::real>& ptable,
            const dev_array1s_ref_t& r, const dev_array2r_ref_t& app) const
         {
         // determine received segment to extract
         const int start = xmax + n * i + x;
         const int length = min(n + dmax, r.size() - start);
         // call batch receiver method
         receiver.R(d, i, r.extract(start, length), ptable);
         // apply priors if applicable
         if (app.size() > 0)
            {
            for (int deltax = -dxmax; deltax <= dxmax; deltax++)
               {
               ptable(dxmax + deltax) *= real(app(i,d));
               }
            }
         }
      //! Compute gamma metric using batch interface, keeping a small local cache
      __device__
      real compute_gamma_batch_cached(int d, int i, int x, int deltax) const
         {
         // NOTE: cuda does not support static in device code, so this requires
         // a recomputation every time, degrading performance to the 'single'
         // interface.
         // space for results
         libcomm::bsid::real ptable_data[libcomm::bsid::metric_computer::arraysize];
         cuda_assertalways(libcomm::bsid::metric_computer::arraysize >= 2 * dxmax + 1);
         cuda::vector_reference<libcomm::bsid::real> ptable(ptable_data, 2 * dxmax + 1);
         // recompute and store every time
         compute_gamma_batch(d, i, x, ptable, r, app);
         // return stored result
         return ptable(dxmax + deltax);
         }
      //! Fill indicated cache entries for gamma metric - batch interface
      __device__
      void fill_gamma_cache_batch(int i, int x) const
         {
         // allocate space for results
         libcomm::bsid::real ptable_data[libcomm::bsid::metric_computer::arraysize];
         cuda_assertalways(libcomm::bsid::metric_computer::arraysize >= 2 * dxmax + 1);
         cuda::vector_reference<libcomm::bsid::real> ptable(ptable_data, 2 * dxmax + 1);
         // get symbol value from thread index
         const int d = threadIdx.x;
         // compute metric with batch interface
         compute_gamma_batch(d, i, x, ptable, r, app);
         // store in corresponding place in cache
         for (int deltax = dmin; deltax <= dmax; deltax++)
            {
            gamma(get_gamma_index(d, i, x, deltax)) = ptable(dxmax + deltax);
            }
         }
      //! Fill indicated cache entries for gamma metric - independent interface
      __device__
      void fill_gamma_cache_single(int i, int x) const
         {
         // get symbol value from thread index
         const int d = threadIdx.x;
         // TODO: no need to compute for all deltax if 'cached' is sufficiently fine
         const int deltaxmin = max(-xmax - x, dmin);
         const int deltaxmax = min(xmax - x, dmax);
         for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
            {
            gamma(get_gamma_index(d, i, x, deltax)) = compute_gamma_single(d, i, x, deltax, r, app);
            }
         }
      //! Wrapper to fill indicated cache entries for gamma metric as needed
      __device__
      void fill_gamma_cache_conditional(int i, int x) const
         {
         // if we not have this already, fill in this part of cache
         if (!cached(i, x + xmax))
            {
            // mark as filled in
            cached(i, x + xmax) = true;
            // call computation method and store results
            if (flags.batch)
               {
               fill_gamma_cache_batch(i, x);
               }
            else
               {
               fill_gamma_cache_single(i, x);
               }
            }
         }
      /*! \brief Wrapper for retrieving gamma metric value
       * This method is called in parallel as follows:
       * - from work_alpha:
       *        d=thread, deltax=block
       *        i=outer loop, x=inner loop
       * - from work_beta:
       *        d=thread, x=block
       *        i=outer loop, deltax=inner loop
       * - from work_message_app:
       *        d=thread, i=block
       *        x=outer loop, deltax=inner loop
       */
      __device__
      real get_gamma(int d, int i, int x, int deltax) const
         {
         // pre-computed values
         if (!flags.lazy)
            {
            return gamma(get_gamma_index(d, i, x, deltax));
            }
         // lazy computation, without global storage
         if (!flags.globalstore)
            {
            if (flags.batch)
               {
               return compute_gamma_batch_cached(d, i, x, deltax);
               }
            else
               {
               return compute_gamma_single(d, i, x, deltax, r, app);
               }
            }
         // lazy computation, with global storage
         fill_gamma_cache_conditional(i, x);
         return gamma(get_gamma_index(d, i, x, deltax));
         }
      // common small tasks
      __device__
      static real get_threshold(const dev_array2r_ref_t& metric, int row, int cols, real factor);
      __device__
      static real parallel_sum(real array[], const int N);
      __device__
      static real get_scale(const dev_array2r_ref_t& metric, int row, int cols);
      __device__
      static void normalize(dev_array2r_ref_t& metric, int row, int cols);
      __device__
      void normalize_alpha(int i)
         {
         normalize(alpha, i, 2 * xmax + 1);
         }
      __device__
      void normalize_beta(int i)
         {
         normalize(beta, i, 2 * xmax + 1);
         }
      // decode functions
      __device__
      void work_gamma_single(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app);
      __device__
      void work_gamma_batch(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app);
      __device__
      void work_alpha(const dev_array1r_ref_t& sof_prior, int i);
      __device__
      void work_beta(const dev_array1r_ref_t& eof_prior, int i);
      __device__
      void work_message_app(dev_array2r_ref_t& ptable) const;
      __device__
      void work_state_app(dev_array1r_ref_t& ptable, const int i) const;
#endif
      // @}
   };
   // @}
private:
   /*! \name Internally-used objects */
   metric_computer computer; //!< Wrapper object for device computation
   dev_array2r_t alpha; //!< Forward recursion metric
   dev_array2r_t beta; //!< Backward recursion metric
   mutable dev_array1r_t gamma; //!< Receiver metric
   mutable dev_array2b_t cached; //!< Flag for globalstore of receiver metric
   dev_array1s_t dev_r; //!< Device copy of received sequence
   dev_array2r_t dev_app; //!< Device copy of a-priori statistics
   dev_array1r_t dev_sof_table; //!< Device copy of sof table
   dev_array1r_t dev_eof_table; //!< Device copy of eof table
   dev_array2r_t dev_ptable; //!< Device copy of results table
   dev_object_t dev_object; //!< Device copy of computer object
   bool initialised; //!< Flag to indicate when memory is allocated
   // @}
private:
   /*! \name Internal functions - main */
   // memory allocation
   void allocate();
   void free();
   // helper methods
   void reset_cache() const;
   void print_gamma(std::ostream& sout) const;
   // data movement
   static void copy_table(const dev_array2r_t& dev_table, array1vr_t& table);
   static void copy_table(const array1vd_t& table, dev_array2r_t& dev_table);
   // decode functions (implemented using kernel calls)
#ifdef __CUDACC__
   void work_gamma(const dev_array1s_t& r, const dev_array2r_t& app);
   void work_alpha(const dev_array1r_t& sof_prior);
   void work_beta(const dev_array1r_t& eof_prior);
   void work_results(dev_array2r_t& ptable, dev_array1r_t& sof_post,
         dev_array1r_t& eof_post) const;
#endif
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2()
      {
      initialised = false;
      }
   // @}

   // main initialization routine
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer, bool norm, bool batch, bool lazy, bool globalstore);

   /*! \name Parameter getters */
   //! Access metric computation
   receiver_t& get_receiver() const
      {
      return computer.receiver;
      }
   int get_N() const
      {
      return computer.N;
      }
   int get_n() const
      {
      return computer.n;
      }
   int get_q() const
      {
      return computer.q;
      }
   int get_I() const
      {
      return computer.I;
      }
   int get_xmax() const
      {
      return computer.xmax;
      }
   int get_dxmax() const
      {
      return computer.dxmax;
      }
   double get_th_inner() const
      {
      return computer.th_inner;
      }
   double get_th_outer() const
      {
      return computer.th_outer;
      }
   // @}

   // decode functions
   void decode(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post, const int offset);

   // Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm [CUDA]";
      }
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
