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
 */

#ifndef __fba2_cuda_h
#define __fba2_cuda_h

#include "fba2-interface.h"
#include "matrix.h"
#include "cuda-all.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

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
 * Implements the forward-backward algorithm for a HMM, as required for the
 * MAP decoding algorithm for a generalized class of synchronization-correcting
 * codes described in
 * Johann A. Briffa, Victor Buttigieg, and Stephan Wesemeyer, "Time-varying
 * block codes for synchronisation errors: maximum a posteriori decoder and
 * practical issues. IET Journal of Engineering, 30 Jun 2014.
 *
 * \warning Do not use shorthand for class hierarchy, as these are not
 * interpreted properly by NVCC.
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 * \tparam thresholding Flag to indicate if we're doing path thresholding
 * \tparam lazy Flag indicating lazy computation of gamma metric
 * \tparam globalstore Flag indicating global pre-computation or caching of gamma values
 */

template <class receiver_t, class sig, class real, class real2,
      bool thresholding, bool lazy, bool globalstore>
class fba2 : public libcomm::fba2_interface<sig, real, real2> {
public:
   // forward definition
   class metric_computer;
public:
   /*! \name Type definitions */
   // Device-based types - data containers
   typedef cuda::vector<int> dev_array1i_t;
   typedef cuda::vector<sig> dev_array1s_t;
   typedef cuda::vector<real> dev_array1r_t;
   typedef cuda::matrix<real> dev_array2r_t;
   typedef cuda::vector<bool> dev_array1b_t;
   typedef cuda::matrix<bool> dev_array2b_t;
   typedef cuda::value<fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer>
         dev_object_t;
   // Device-based types - references
   typedef cuda::vector_reference<sig> dev_array1s_ref_t;
   typedef cuda::vector_reference<real> dev_array1r_ref_t;
   typedef cuda::matrix_reference<real> dev_array2r_ref_t;
   typedef cuda::vector_reference<bool> dev_array1b_ref_t;
   typedef cuda::matrix_reference<bool> dev_array2b_ref_t;
   typedef cuda::value_reference<fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer>
         dev_object_ref_t;
   // Host-based types
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
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
         dev_array2r_ref_t global; //!< global storage: indices (i,[x,deltax,d])
         dev_array2r_ref_t local; //!< local storage: indices (i%depth,[x,deltax,d])
      } gamma; //!< Receiver metric
      mutable struct {
         dev_array2b_ref_t global; //!< global storage: indices (i,x)
         dev_array2b_ref_t local; //!< local storage: indices (i%depth,x)
      } cached; //!< Flag for caching of receiver metric
      mutable dev_array1i_t cw_length; //!< Codeword 'i' length
      mutable dev_array1i_t cw_start; //!< Codeword 'i' start
      mutable int tau; //!< Frame length (all codewords in sequence)
      dev_array1s_ref_t r; //!< Copy of received sequence, for lazy or local computation of gamma
      dev_array2r_ref_t app; //!< Copy of a-priori statistics, for lazy or local computation of gamma
      // @}
      /*! \name User-defined parameters */
      real th_inner; //!< Threshold factor for computing alpha/beta
      real th_outer; //!< Threshold factor for computing message APPs
      int N; //!< The transmitted block size in symbols
      int q; //!< The number of symbols in the q-ary alphabet
      int mtau_min; //!< The largest negative drift within a whole frame is \f$ m_\tau^{-} \f$
      int mtau_max; //!< The largest positive drift within a whole frame is \f$ m_\tau^{+} \f$
      int mn_min; //!< The largest negative drift within a q-ary symbol is \f$ m_n^{-} \f$
      int mn_max; //!< The largest positive drift within a q-ary symbol is \f$ m_n^{+} \f$
      int m1_min; //!< The largest negative drift over a single channel symbol is \f$ m_1^{-} \f$
      int m1_max; //!< The largest positive drift over a single channel symbol is \f$ m_1^{+} \f$
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
         //cuda_assert(d >= 0 && d < q);
         //cuda_assert(x >= mtau_min && x <= mtau_max);
         //cuda_assert(deltax >= mn_min && deltax <= mn_max);
         // determine index to use
         int ndx = 0;
         /* gamma needs indices:
          *   (x,deltax,d)
          * where:
          *   x in [mtau_min, mtau_max]
          *   d in [0, q-1]
          *   deltax in [mn_min, mn_max]
          */
         ndx = (x - mtau_min);
         ndx = (deltax - mn_min) + (mn_max - mn_min + 1) * ndx;
         ndx = d + q * ndx;
         // host code path only
#ifndef __CUDA_ARCH__
#if DEBUG>=2
         std::cerr << "(" << d << "," << x << "," << deltax << ":" << ndx << ")";
#endif
#endif
         return ndx;
         }
      //! Get a reference to the corresponding gamma storage entry
      __device__
      real& gamma_storage_entry(int d, int i, int x, int deltax) const
         {
         if (globalstore)
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
         cuda_assertalways(arraysize >= mn_max - mn_min + 1);
         cuda::vector_reference<real2> ptable(ptable_data, mn_max - mn_min + 1);
         // determine received segment to extract
         const int start = cw_start(i) + x - mtau_min;
         const int length = min(cw_length(i) + mn_max, r.size() - start);
         // get symbol value from thread index
         for(int d = threadIdx.x; d < q; d += blockDim.x)
            {
            // call batch receiver method
            receiver.R(d, i, r.extract(start, length), ptable);
            // apply priors if applicable
            if (app.size() > 0)
               for (int deltax = mn_min; deltax <= mn_max; deltax++)
                  ptable(deltax - mn_min) *= real(app(i,d));
            // store in corresponding place in cache
            for (int deltax = mn_min; deltax <= mn_max; deltax++)
               gamma_storage_entry(d, i, x, deltax) = ptable(deltax - mn_min);
            }
         }
      /*! \brief Fill indicated cache entries for gamma metric as needed
       *
       * This method is called on every get_gamma call when doing lazy computation.
       * It will update the cache as needed, for both local/global storage.
       */
      __device__
      void fill_gamma_cache_conditional(int i, int x) const
         {
         bool miss = false;
         if (globalstore)
            {
            // if we not have this already, mark to fill in this part of cache
            if (!cached.global(i, x - mtau_min))
               {
               miss = true;
               cached.global(i, x - mtau_min) = true;
               }
            }
         else
            {
            // if we not have this already, mark to fill in this part of cache
            if (!cached.local(i & (depth-1), x - mtau_min))
               {
               miss = true;
               cached.local(i & (depth-1), x - mtau_min) = true;
               }
            }
         if (miss)
            {
            // call computation method and store results
            fill_gamma_storage_batch(r, app, i, x);
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
         if (lazy)
            fill_gamma_cache_conditional(i, x);
         return gamma_storage_entry(d, i, x, deltax);
         }
      // common small tasks
      __device__
      static real get_threshold(const real metric[], int imin, int imax, real factor)
         {
         // early short-cut for no-thresholding
         if (!thresholding || factor == 0)
            return 0;
         // actual computation
         real threshold = 0;
         for (int i = imin; i <= imax; i++)
            if (metric[i] > threshold)
               threshold = metric[i];
         return threshold * factor;
         }
      //! Copy slice of metric (row) to shared memory in parallel
      __device__
      static void parallel_copy_slice(real dst[], const dev_array2r_ref_t& src, int i)
         {
         cuda_assert(i >= 0);
         // copy required slice from main memory (in parallel)
         for (int j = threadIdx.x; j < src.get_cols(); j += blockDim.x)
            dst[j] = src(i, j);
         // make sure all threads have done their bit
         __syncthreads();
         }
      //! Reset values in array in parallel
      __device__
      static void parallel_reset_array(real x[], int n)
         {
         // reset all array entries to zero (in parallel)
         for(int i = threadIdx.x; i < n; i += blockDim.x)
            x[i] = 0;
         // make sure all threads have done their bit
         __syncthreads();
         }
      /*! \brief Computes the sum of the elements in the given array (length N).
       * The sum is computed in parallel between the threads in a given block,
       * and also returned as the first element in the array.
       * \note A limitation for this to work is that N must be a power of 2,
       * and the block size has to be at least N/2 threads.
       * \warning The contents of the array are destroyed in the process.
       */
      __device__
      static void parallel_sum(real array[], const int N)
         {
         const int i = threadIdx.x;
         cuda_assert(N / 2 <= blockDim.x); // Total number of active threads
         for(int n = N; n > 1; n >>= 1)
            {
            const int half = (n >> 1); // divide by two
            cuda_assert(2 * half == n);
            // only the first half of the threads will be active.
            if (i < half)
               array[i] += array[i + half];
            // wait until all threads have completed their part
            __syncthreads();
            }
         }
      __device__
      static real get_scale(const dev_array2r_ref_t& metric, int row, int cols)
         {
         real scale = 0;
         for (int col = 0; col < cols; col++)
            scale += metric(row, col);
         cuda_assertalways(scale > real(0));
         scale = real(1) / scale;
         return scale;
         }
      __device__
      static void normalize(dev_array2r_ref_t& metric, int row, int cols)
         {
         // set up thread index
         const int col = threadIdx.x;
         cuda_assert(col < cols);
         // determine the scale factor to use (each thread has to do this)
         const real scale = get_scale(metric, row, cols);
         // wait until all threads have completed the computation
         __syncthreads();
         // scale all results
         metric(row, col) *= scale;
         }
      __device__
      void normalize_alpha(int i)
         {
         normalize(alpha, i, mtau_max - mtau_min + 1);
         }
      __device__
      void normalize_beta(int i)
         {
         normalize(beta, i, mtau_max - mtau_min + 1);
         }
      // decode functions - partial computations
      __device__
      void work_gamma(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app, const int i, const int x) const
         {
         fill_gamma_storage_batch(r, app, i, x);
         }
      __device__
      void init_alpha(const dev_array1r_ref_t& sof_prior);
      __device__
      void init_beta(const dev_array1r_ref_t& eof_prior);
      __device__
      void work_alpha(const int i, const int x2);
      __device__
      void work_beta(const int i, const int x1);
      __device__
      void work_message_app(dev_array2r_ref_t& ptable, const int i, const int d) const;
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
      dev_array2r_t global; //!< global storage: indices (i,[x,deltax,d])
      dev_array2r_t local; //!< local storage: indices (i%depth,[x,deltax,d])
   } gamma; //!< Receiver metric
   mutable struct {
      dev_array2b_t global; //!< global storage: indices (i,x)
      dev_array2b_t local; //!< local storage: indices (i%depth,x)
   } cached; //!< Flag for caching of receiver metric
   dev_array1s_t dev_r; //!< Device copy of received sequence
   dev_array2r_t dev_app; //!< Device copy of a-priori statistics
   dev_array1r_t dev_sof_table; //!< Device copy of sof table
   dev_array1r_t dev_eof_table; //!< Device copy of eof table
   dev_array2r_t dev_ptable; //!< Device copy of results table
   dev_object_t dev_object; //!< Device copy of computer object
   bool initialised; //!< Flag to indicate when memory is allocated
   int gamma_thread_count; //!< Number of threads per block for gamma kernel
   int alpha_thread_count; //!< Number of threads per block for alpha kernel
   int beta_thread_count; //!< Number of threads per block for beta kernel
   int message_thread_count; //!< Number of threads per block for message app kernel
   // @}
private:
   /*! \name Internal functions - main */
   // memory allocation
   void allocate();
   void free();
   // helper methods
   void reset_cache() const;
   void print_gamma(std::ostream& sout) const;
#ifdef __CUDACC__
   //! Store the resource parameters for given kernel
   template <class T>
   static void add_kernel_parameters(libcomm::instrumented& collector,
         const std::string& name, const dim3& gridDim, const dim3& blockDim,
         const size_t sharedMem, T* entry)
      {
      // shorthand
      const void* fptr = (const void*) entry;
      const int max_threads = cudaGetMaxThreadsPerBlock(fptr);
      const int regs = cudaGetNumRegsPerThread(fptr);
      const size_t lmem = cudaGetLocalSize(fptr);
      const size_t smem = cudaGetSharedSize(fptr);
      // add the required parameters to collector
      collector.add_timer(count(gridDim), "k_" + name + "_blocks");
      collector.add_timer(count(blockDim), "k_" + name + "_threads");
      collector.add_timer(regs, "k_" + name + "_regs");
      collector.add_timer(lmem, "k_" + name + "_lmem");
      collector.add_timer(smem + sharedMem, "k_" + name + "_smem");
#ifndef NDEBUG
      // inform user what the kernel sizes and resources are
      std::cerr << name << ": " << gridDim << " x " << blockDim;
      std::cerr << ", max_threads = " << max_threads;
      std::cerr << ", regs = " << regs;
      if (lmem)
         std::cerr << ", lmem = " << lmem;
      if (sharedMem + smem)
         std::cerr << ", smem = " << smem << "+" << sharedMem;
      std::cerr << std::endl;
#endif
      }
#endif
   /*! \brief Determine the thread count to use for given kernel.
    * \param preferred The requested (largest wanted) block size in threads
    * \param smem_constant The base amount of dynamic shared memory needed per block
    * \param smem_variable The additional amount of dynamic shared memory needed per group-of-threads (or part thereof)
    * \param group_size The group-of-threads size
    * \param entry The kernel function entry point
    * \return Number of threads per block to use
    *
    * This method determines the number of threads per block to use for a given
    * kernel, respecting the maximum thread count for the device+kernel
    * combination. This takes into account register pressure and device limits.
    * The methods tries to choose a number of threads that is a multiple of
    * the warp size.
    */
   template <class T>
   static int determine_thread_count(int preferred, size_t smem_constant,
         size_t smem_variable, int group_size, T* entry)
      {
      // shorthand
      const void* fptr = (const void*) entry;
      // get kernel and device parameters
      const int max_threads = cudaGetMaxThreadsPerBlock(fptr);
      const int warp_size = cudaGetWarpSize();
      const size_t smem_per_block = smem_constant + cudaGetSharedSize(fptr);
      const size_t max_smem = cudaGetSharedMemPerBlock();
      // start with the largest number of warps the device will take
      int thread_count = std::min(preferred, max_threads);
      int warps = (thread_count + warp_size - 1) / warp_size;
      int groups = (thread_count + group_size - 1) / group_size;
      // find the largest multiple of warps that will fit
      while (smem_per_block + smem_variable * groups > max_smem)
         {
         warps -= 1;
         thread_count = warps * warp_size;
         groups = (thread_count + group_size - 1) / group_size;
         }
      assertalways(thread_count > 0);
      return thread_count;
      }
   // data movement
   static void copy_table(const dev_array2r_t& dev_table, array1vr_t& table);
   static void copy_table(const array1vd_t& table, dev_array2r_t& dev_table);
   // device setup function
   void setup(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, const int offset);
   void setup_gamma(libcomm::instrumented& collector);
   void transfer(libcomm::instrumented& collector, array1vr_t& ptable,
         array1r_t& sof_post, array1r_t& eof_post);
#ifdef __CUDACC__
   // decode functions - global path (implemented using kernel calls)
   void work_gamma(libcomm::instrumented& collector, const dev_array1s_t& r,
         const dev_array2r_t& app);
   void work_alpha_and_beta(libcomm::instrumented& collector,
         const dev_array1r_t& sof_prior, const dev_array1r_t& eof_prior);
   void work_results(libcomm::instrumented& collector, dev_array2r_t& ptable,
         dev_array1r_t& sof_post, dev_array1r_t& eof_post) const;
   // decode functions - local path (implemented using kernel calls)
   void work_alpha(libcomm::instrumented& collector,
         const dev_array1r_t& sof_prior);
   void work_beta_and_results(libcomm::instrumented& collector,
         const dev_array1r_t& eof_prior, dev_array2r_t& ptable,
         dev_array1r_t& sof_post, dev_array1r_t& eof_post);
#endif
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2() :
      initialised(false)
      {
      }
   // @}

   /*! \name FBA2 Interface Implementation */
   /*! \brief Set up code size, decoding parameters, and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(int N, int q, int mtau_min, int mtau_max, int mn_min, int mn_max,
         int m1_min, int m1_max, double th_inner, double th_outer,
         const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer);
   /*! \brief Set up encoding table
    * Needs to be done before every frame.
    */
   void init(const array2vs_t& encoding_table) const
      {
      // Initialize arrays with start and length of each codeword
      array1i_t cw_length; //!< Codeword 'i' length
      array1i_t cw_start; //!< Codeword 'i' start
      cw_length.init(computer.N);
      cw_start.init(computer.N);
      int start = 0;
      for (int i = 0; i < computer.N; i++)
         {
         const int n = encoding_table(i, 0).size();
         cw_start(i) = start;
         cw_length(i) = n;
         start += n;
         }
      // Transfer to device
      this->computer.cw_start = cw_start;
      this->computer.cw_length = cw_length;
      this->computer.tau = start;
      // Set up receiver with new encoding table
      this->computer.receiver.init(encoding_table);
      }

   // decode functions
   void decode(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post, const int offset);
   void get_drift_pdf(array1r_t& pdf, const int i) const;
   void get_drift_pdf(array1vr_t& pdftable) const;

   //! Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm [CUDA]";
      }
   // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
