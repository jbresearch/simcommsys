/*!
 * \file
 * $Id$
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
 */

#ifndef __fba2_cuda_h
#define __fba2_cuda_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "fsm.h"
#include "cuda-all.h"
#include "instrumented.h"

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
 * $Id$
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * MAP decoding algorithm for a generalized class of synchronization-correcting
 * codes described in
 * Briffa et al, "A MAP Decoder for a General Class of Synchronization-
 * Correcting Codes", Submitted to Trans. IT, 2011.
 *
 * \warning Do not use shorthand for class hierarchy, as these are not
 * interpreted properly by NVCC.
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 */

template <class receiver_t, class sig, class real, class real2>
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
   typedef cuda::vector<bool> dev_array1b_t;
   typedef cuda::matrix<bool> dev_array2b_t;
   typedef cuda::value<fba2<receiver_t, sig, real, real2>::metric_computer>
         dev_object_t;
   // Device-based types - references
   typedef cuda::vector_reference<sig> dev_array1s_ref_t;
   typedef cuda::vector_reference<real> dev_array1r_ref_t;
   typedef cuda::matrix_reference<real> dev_array2r_ref_t;
   typedef cuda::vector_reference<bool> dev_array1b_ref_t;
   typedef cuda::matrix_reference<bool> dev_array2b_ref_t;
   typedef cuda::value_reference<fba2<receiver_t, sig, real, real2>::metric_computer>
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
      mutable struct {
         dev_array2r_ref_t global; // indices (i,x,d,deltax)
         dev_array2r_ref_t local; // indices (i%depth,x,d,deltax)
      } gamma; //!< Receiver metric
      mutable struct {
         dev_array2b_ref_t global; // indices (i,x)
         dev_array2b_ref_t local; // indices (i%depth,x)
      } cached; //!< Flag for caching of receiver metric
      dev_array1s_ref_t r; //!< Copy of received sequence, for lazy or local computation of gamma
      dev_array2r_ref_t app; //!< Copy of a-priori statistics, for lazy or local computation of gamma
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
         bool globalstore; //!< Flag indicating global pre-computation or caching of gamma values
      } flags;
      // @}
      /*! \name Hardwired parameters */
      static const int arraysize = 2 * 63 + 1; //!< Size of stack-allocated arrays
      static const int depth = 4; //!< Number of indices that can be processed simultaneously (must be a power of 2)
      // @}
   public:
      /*! \name Internal functions - computer */
#ifdef __CUDACC__
      __device__ __host__
      int get_gamma_index(int d, int x, int deltax) const
         {
         cuda_assert(d >= 0 && d < q);
         cuda_assert(x >= -xmax && x <= xmax);
         cuda_assert(deltax >= dmin && deltax <= dmax);
         // determine index to use
         int ndx = 0;
         /* gamma needs indices:
          *   (x,deltax,d)
          * where:
          *   x in [-xmax, xmax]
          *   d in [0, q-1]
          *   deltax in [dmin, dmax]
          */
         const int pitch1 = q;
         const int pitch2 = pitch1 * (dmax - dmin + 1);
         ndx += pitch2 * (x + xmax);
         ndx += pitch1 * (deltax - dmin);
         ndx += d;
         // host code path only
#ifndef __CUDA_ARCH__
#if DEBUG>=2
         std::cerr << "(" << d << "," << x << "," << deltax << ":" << ndx << ")";
#endif
#endif
         return ndx;
         }
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
            result *= real(app(i,d));
         return result;
         }
      //! Compute gamma metric using batch receiver interface
      __device__
      void compute_gamma_batch(int d, int i, int x, vector_reference<real2>& ptable,
            const dev_array1s_ref_t& r, const dev_array2r_ref_t& app) const
         {
         // determine received segment to extract
         const int start = xmax + n * i + x;
         const int length = min(n + dmax, r.size() - start);
         // call batch receiver method
         receiver.R(d, i, r.extract(start, length), ptable);
         // apply priors if applicable
         if (app.size() > 0)
            for (int deltax = -dxmax; deltax <= dxmax; deltax++)
               ptable(dxmax + deltax) *= real(app(i,d));
         }
      //! Get a reference to the corresponding gamma storage entry
      __device__
      real& gamma_storage_entry(int d, int i, int x, int deltax) const
         {
         if (flags.globalstore)
            return gamma.global(i, get_gamma_index(d, x, deltax));
         else
            return gamma.local(i & (depth-1), get_gamma_index(d, x, deltax));
         }
      //! Fill indicated storage entries for gamma metric - batch interface
      __device__
      void fill_gamma_storage_batch(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app, int i, int x) const
         {
         // allocate space for results
         real2 ptable_data[arraysize];
         cuda_assertalways(arraysize >= 2 * dxmax + 1);
         cuda::vector_reference<real2> ptable(ptable_data, 2 * dxmax + 1);
         // get symbol value from thread index
         for(int d = threadIdx.x; d < q; d += blockDim.x)
            {
            // compute metric with batch interface
            compute_gamma_batch(d, i, x, ptable, r, app);
            // store in corresponding place in cache
            for (int deltax = dmin; deltax <= dmax; deltax++)
               gamma_storage_entry(d, i, x, deltax) = ptable(dxmax + deltax);
            }
         }
      /*! \brief Fill indicated storage entries for gamma metric - independent interface
       * \todo No need to compute for all deltax if 'cached' is sufficiently fine
       */
      __device__
      void fill_gamma_storage_single(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app, int i, int x) const
         {
         // get symbol value from thread index
         for(int d = threadIdx.x; d < q; d += blockDim.x)
            {
            // clear gamma entries
            for (int deltax = dmin; deltax <= dmax; deltax++)
               gamma_storage_entry(d, i, x, deltax) = 0;
            // limit on end-state (-xmax <= x2 <= xmax):
            //   x2-x1 <= xmax-x1
            //   x2-x1 >= -xmax-x1
            const int deltaxmin = max(-xmax - x, dmin);
            const int deltaxmax = min(xmax - x, dmax);
            // compute entries within required limits
            for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
               gamma_storage_entry(d, i, x, deltax) = compute_gamma_single(d, i, x, deltax, r, app);
            }
         }
      /*! \brief Fill indicated cache entries for gamma metric as needed
       *
       * This method is called on every get_gamma call when doing lazy computation.
       * It will update the cache as needed, for both local/global storage,
       * and choosing between batch/single methods as required.
       */
      __device__
      void fill_gamma_cache_conditional(int i, int x) const
         {
         bool miss = false;
         if (flags.globalstore)
            {
            // if we not have this already, mark to fill in this part of cache
            if (!cached.global(i, x + xmax))
               {
               miss = true;
               cached.global(i, x + xmax) = true;
               }
            }
         else
            {
            // if we not have this already, mark to fill in this part of cache
            if (!cached.local(i & (depth-1), x + xmax))
               {
               miss = true;
               cached.local(i & (depth-1), x + xmax) = true;
               }
            }
         if (miss)
            {
            // call computation method and store results
            if (flags.batch)
               fill_gamma_storage_batch(r, app, i, x);
            else
               fill_gamma_storage_single(r, app, i, x);
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
         // update cache values if necessary
         if (flags.lazy)
            fill_gamma_cache_conditional(i, x);
         return gamma_storage_entry(d, i, x, deltax);
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
      // decode functions - partial computations
      __device__
      void work_gamma(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app, const int i) const
         {
         // get start drift from block index
         const int x = blockIdx.y - xmax;
         if (flags.batch)
            fill_gamma_storage_batch(r, app, i, x);
         else
            fill_gamma_storage_single(r, app, i, x);
         }
      __device__
      void init_alpha(const dev_array1r_ref_t& sof_prior);
      __device__
      void init_beta(const dev_array1r_ref_t& eof_prior);
      __device__
      void work_alpha(const int i);
      __device__
      void work_beta(const int i);
      __device__
      void work_message_app(dev_array2r_ref_t& ptable, const int i) const;
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
   mutable struct {
      dev_array2r_t global; // indices (i,x,d,deltax)
      dev_array2r_t local; // indices (i%depth,x,d,deltax)
   } gamma; //!< Receiver metric
   mutable struct {
      dev_array2b_t global; // indices (i,x)
      dev_array2b_t local; // indices (i%depth,x)
   } cached; //!< Flag for caching of receiver metric
   dev_array1s_t dev_r; //!< Device copy of received sequence
   dev_array2r_t dev_app; //!< Device copy of a-priori statistics
   dev_array1r_t dev_sof_table; //!< Device copy of sof table
   dev_array1r_t dev_eof_table; //!< Device copy of eof table
   dev_array2r_t dev_ptable; //!< Device copy of results table
   dev_object_t dev_object; //!< Device copy of computer object
   bool initialised; //!< Flag to indicate when memory is allocated
   /*! \name Hardwired parameters */
   static const int max_threads = 1024; //!< Maximum number of threads per block (ideally a multiple of the warp size)
   // @}
private:
   /*! \name Internal functions - main */
   // memory allocation
   void allocate();
   void free();
   // helper methods
   void reset_cache() const;
   void print_gamma(std::ostream& sout) const;
   int get_gamma_threadcount() const
      {
      if (computer.q < max_threads)
         return computer.q;
      return max_threads;
      }
   // data movement
   static void copy_table(const dev_array2r_t& dev_table, array1vr_t& table);
   static void copy_table(const array1vd_t& table, dev_array2r_t& dev_table);
#ifdef __CUDACC__
   // decode functions - global path (implemented using kernel calls)
   void work_gamma(const dev_array1s_t& r, const dev_array2r_t& app);
   void work_alpha_and_beta(const dev_array1r_t& sof_prior,
         const dev_array1r_t& eof_prior);
   void work_results(dev_array2r_t& ptable, dev_array1r_t& sof_post,
         dev_array1r_t& eof_post) const;
   // decode functions - local path (implemented using kernel calls)
   void work_alpha(const dev_array1r_t& sof_prior);
   void work_beta_and_results(const dev_array1r_t& eof_prior,
         dev_array2r_t& ptable, dev_array1r_t& sof_post, dev_array1r_t& eof_post);
#endif
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2() :
      initialised(false)
      {
      }
   /*! \brief Copy constructor
    * \note Copy construction is a deep copy, except for:
    *       - metric tables (alpha, beta, gamma, cached), on device
    *       - copies of received and prior statistics, on device
    *       - copies of sof/eof and results tables, on device
    *       - copy of computer object on device
    *
    * \todo This will not be necessary (can keep the default copy constructor)
    *       when/if the TX and RX side of commsys objects are separated, as we
    *       won't need to clone the RX commsys object in stream simulations.
    */
   fba2(const fba2<receiver_t, sig, real, real2>& x) :
      computer(x.computer), initialised(false)
      {
      }
   // @}

   // main initialization routine
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer, bool norm, bool batch, bool lazy, bool globalstore);

   /*! \name Parameter getters */
   //! Determine memory required for global storage mode (in MiB)
   static int get_memory_required(int N, int n, int q, int I, int xmax,
         int dxmax)
      {
      // determine allowed limits on deltax: see allocate() for documentation
      const int dmin = std::max(-n, -dxmax);
      const int dmax = std::min(n * I, dxmax);
      // determine memory required
      // NOTE: do all computations at 64-bit, or we get intermediate overflow!
      libbase::int64u bytes_required = sizeof(real);
      bytes_required *= q;
      bytes_required *= N;
      bytes_required *= (2 * xmax + 1);
      bytes_required *= (dmax - dmin + 1);
      bytes_required >>= 20;
      return int(bytes_required);
      }
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
   void get_drift_pdf(array1r_t& pdf, const int i) const;
   void get_drift_pdf(array1vr_t& pdftable) const;

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
