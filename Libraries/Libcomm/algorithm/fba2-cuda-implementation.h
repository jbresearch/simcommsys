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

#include "fba2-cuda.h"
#include "pacifier.h"
#include "vectorutils.h"
#include "cputimer.h"
#include "cuda/stream.h"
#include "cuda/event.h"
#include "cuda/util.h"
#include <iomanip>

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show allocated memory sizes
// 3 - Show input and intermediate vectors when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// *** Metric Computer ***

// Internal procedures

// decode functions - partial computations

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::init_alpha(const dev_array1r_ref_t& sof_prior)
   {
   // get end drift from thread index
   const int x2 = threadIdx.x + mtau_min;

   // set array initial conditions (parallelized):
   // set initial drift distribution
   alpha(0, x2 - mtau_min) = sof_prior(x2 - mtau_min);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::init_beta(const dev_array1r_ref_t& eof_prior)
   {
   // get start drift from thread index
   const int x1 = threadIdx.x + mtau_min;

   // set array initial conditions (parallelized):
   // set final drift distribution
   beta(N, x1 - mtau_min) = eof_prior(x1 - mtau_min);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::work_alpha(const int i, const int x2)
   {
   using cuda::min;
   using cuda::max;

   // set up variables shared within block
   SharedMemory<real> smem;
   const int pitch = q + (mtau_max - mtau_min + 1);
   __restrict__ real* this_alpha = smem.getPointer() + threadIdx.y * pitch;
   __restrict__ real* alpha_slice = this_alpha + q - mtau_min;
   // initialization
   cuda_assert(i > 0);
   parallel_copy_slice(alpha_slice + mtau_min, alpha, i - 1);
   parallel_reset_array(this_alpha, q);

   // compute remaining matrix values:

   // determine the strongest path at this point
   const real threshold = get_threshold(alpha_slice, mtau_min, mtau_max, th_inner);
   // get symbol value from thread index
   for(int d = threadIdx.x; d < q; d += blockDim.x)
      {
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= mn_max
      //   x2-x1 >= mn_min
      const int x1min = max(mtau_min, x2 - mn_max);
      const int x1max = min(mtau_max, x2 - mn_min);
      for (int x1 = x1min; x1 <= x1max; x1++)
         {
         // ignore paths below a certain threshold
         if (!thresholding || alpha_slice[x1] >= threshold)
            {
            // each block computes for a different end-state (x2)
            // each thread in a block is computing for a different symbol (d)
            this_alpha[d] += alpha_slice[x1] * get_gamma(d, i - 1, x1, x2 - x1);
            }
         }
      }
   // make sure all threads in block have finished updating this_alpha
   __syncthreads();
   // compute sum of shared array
   parallel_sum(this_alpha, q);
   // store result (first thread in block)
   if (threadIdx.x == 0)
      alpha(i, x2 - mtau_min) = this_alpha[0];
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::work_beta(const int i, const int x1)
   {
   using cuda::min;
   using cuda::max;

   // set up variables shared within block
   SharedMemory<real> smem;
   const int pitch = q + (mtau_max - mtau_min + 1);
   __restrict__ real* this_beta = smem.getPointer() + threadIdx.y * pitch;
   __restrict__ real* beta_slice = this_beta + q - mtau_min;
   // initialization
   cuda_assert(i < N);
   parallel_copy_slice(beta_slice + mtau_min, beta, i + 1);
   parallel_reset_array(this_beta, q);

   // compute remaining matrix values:

   // determine the strongest path at this point
   const real threshold = get_threshold(beta_slice, mtau_min, mtau_max, th_inner);
   // get symbol value from thread index
   for(int d = threadIdx.x; d < q; d += blockDim.x)
      {
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= mn_max
      //   x2-x1 >= mn_min
      const int x2min = max(mtau_min, mn_min + x1);
      const int x2max = min(mtau_max, mn_max + x1);
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         // ignore paths below a certain threshold
         if (!thresholding || beta_slice[x2] >= threshold)
            {
            // each block computes for a different start-state (x1)
            // each thread in a block is computing for a different symbol (d)
            this_beta[d] += beta_slice[x2] * get_gamma(d, i, x1, x2 - x1);
            }
         }
      }
   // make sure all threads in block have finished updating this_beta
   __syncthreads();
   // compute sum of shared array
   parallel_sum(this_beta, q);
   // store result (first thread in block)
   if (threadIdx.x == 0)
      beta(i, x1 - mtau_min) = this_beta[0];
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::work_message_app(dev_array2r_ref_t& ptable, const int i, const int d) const
   {
   using cuda::min;
   using cuda::max;

   // get access to alpha and beta slices in shared memory
   SharedMemory<real> smem;
   real *alpha_slice = smem.getPointer() - mtau_min;
   real *beta_slice = smem.getPointer() + (mtau_max - mtau_min + 1) - mtau_min;
   // copy required slices from main memory (in parallel)
   parallel_copy_slice(alpha_slice + mtau_min, alpha, i);
   parallel_copy_slice(beta_slice + mtau_min, beta, i + 1);

   // Check result vector (one symbol per timestep)
   cuda_assertalways(ptable.get_rows()==N && ptable.get_cols()==q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   // - all threads are independent and indexes guaranteed in range
   // determine the strongest path at this point
   const real threshold = get_threshold(alpha_slice, mtau_min, mtau_max, th_inner);
   // initialize result holder
   real p = 0;
   for (int x1 = mtau_min; x1 <= mtau_max; x1++)
      {
      // ignore paths below a certain threshold
      if (!thresholding || alpha_slice[x1] >= threshold)
         {
         // limits on deltax can be combined as (c.f. allocate() for details):
         //   x2-x1 <= mn_max
         //   x2-x1 >= mn_min
         const int x2min = max(mtau_min, mn_min + x1);
         const int x2max = min(mtau_max, mn_max + x1);
         real temp = 0;
         for (int x2 = x2min; x2 <= x2max; x2++)
            temp += beta_slice[x2] * get_gamma(d, i, x1, x2 - x1);
         p += temp * alpha_slice[x1];
         }
      }
   // store result
   ptable(i,d) = p;
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__device__
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer::work_state_app(dev_array1r_ref_t& ptable,
      const int i) const
   {
   // Check result vector and requested index
   cuda_assertalways(ptable.size()==mtau_max-mtau_min+1);
   cuda_assert(i >= 0 && i <= N);
   // set up block & thread indexes
   const int x = threadIdx.x + mtau_min;
   // compute posterior probabilities for given index
   ptable(x - mtau_min) = alpha(i, x - mtau_min) * beta(i, x - mtau_min);
   }

// *** Kernels *** NOTE: all kernels *must* be global functions

// common small tasks

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_normalize_alpha_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const int i)
   {
   object().normalize_alpha(i);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_normalize_beta_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const int i)
   {
   object().normalize_beta(i);
   }

// decode functions - partial computations

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_gamma_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const vector_reference<sig> r,
      const matrix_reference<real> app, const int i)
   {
   // get start drift from combination of block and thread indices
   const int x = (blockIdx.y * blockDim.y + threadIdx.y) + object().mtau_min;
   if (x <= object().mtau_max)
      object().work_gamma(r, app, i, x);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_init_alpha_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const vector_reference<real> sof_prior)
   {
   object().init_alpha(sof_prior);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_init_beta_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const vector_reference<real> eof_prior)
   {
   object().init_beta(eof_prior);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_alpha_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const int i)
   {
   // get end drift from combination of block and thread indices
   const int x2 = (blockIdx.x * blockDim.y + threadIdx.y) + object().mtau_min;
   if (x2 <= object().mtau_max)
      object().work_alpha(i, x2);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_beta_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const int i)
   {
   // get start drift from combination of block and thread indices
   const int x1 = (blockIdx.x * blockDim.y + threadIdx.y) + object().mtau_min;
   if (x1 <= object().mtau_max)
      object().work_beta(i, x1);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_message_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, matrix_reference<real> ptable, const int i)
   {
   // get symbol value from thread index
   const int d = threadIdx.x + blockDim.x * blockIdx.y;
   if (d < object().q)
      object().work_message_app(ptable, i, d);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_state_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, vector_reference<real> ptable, const int i)
   {
   object().work_state_app(ptable, i);
   }

// decode functions - global path

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_global_gamma_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, const vector_reference<sig> r,
      const matrix_reference<real> app)
   {
   // get symbol index from block index
   const int i = blockIdx.x;
   // get start drift from combination of block and thread indices
   const int x = (blockIdx.y * blockDim.y + threadIdx.y) + object().mtau_min;
   if (x <= object().mtau_max)
      object().work_gamma(r, app, i, x);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
__global__
void fba2_global_message_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::metric_computer> object, matrix_reference<real> ptable)
   {
   // get symbol index from block index
   const int i = blockIdx.x;
   // get symbol value from thread index
   const int d = threadIdx.x + blockDim.x * blockIdx.y;
   if (d < object().q)
      object().work_message_app(ptable, i, d);
   }

// *** Main Class

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::allocate()
   {
   // flag the state of the arrays
   initialised = true;
   // shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int mtau_min = computer.mtau_min;
   const int mtau_max = computer.mtau_max;
   const int mn_min = computer.mn_min;
   const int mn_max = computer.mn_max;

   // alpha needs indices (i,x) where i in [0, N] and x in [mtau_min, mtau_max]
   // beta needs indices (i,x) where i in [0, N] and x in [mtau_min, mtau_max]
   alpha.init(N + 1, mtau_max - mtau_min + 1); // offsets: 0, -mtau_min
   beta.init(N + 1, mtau_max - mtau_min + 1); // offsets: 0, -mtau_min

   if (globalstore)
      {
      /* gamma needs indices (i,x,d,deltax) where
       * i in [0, N-1]
       * x in [mtau_min, mtau_max]
       * d in [0, q-1]
       * deltax in [mn_min, mn_max]
       * NOTE: this is allocated as a flat sequence
       */
      gamma.global.init(N,
            (mtau_max - mtau_min + 1) * q * (mn_max - mn_min + 1));
      gamma.local.init(0, 0);
      }
   else
      {
      /* gamma needs indices (x,d,deltax) where
       * x in [mtau_min, mtau_max]
       * d in [0, q-1]
       * deltax in [mn_min, mn_max]
       * NOTE: this is allocated as a flat sequence
       */
      gamma.local.init(computer.depth,
            (mtau_max - mtau_min + 1) * q * (mn_max - mn_min + 1));
      gamma.global.init(0, 0);
      }
   // need to keep track only if we're caching lazy computations
   if (lazy)
      {
      if (globalstore)
         {
         /* cached needs indices (i,x) where
          * i in [0, N-1]
          * x in [mtau_min, mtau_max]
          */
         cached.global.init(N, mtau_max - mtau_min + 1); // offsets: 0, -mtau_min
         cached.local.init(0, 0);
         }
      else
         {
         /* cached needs indices (x) where
          * x in [mtau_min, mtau_max]
          */
         cached.local.init(computer.depth, mtau_max - mtau_min + 1); // offsets: 0, -mtau_min
         cached.global.init(0, 0);
         }
      }
   else
      {
      cached.global.init(0, 0);
      cached.local.init(0, 0);
      }
   // copy over to references
   computer.alpha = alpha;
   computer.beta = beta;
   computer.gamma.global = gamma.global;
   computer.gamma.local = gamma.local;
   computer.cached.global = cached.global;
   computer.cached.local = cached.local;

   // if this is not the first time, skip the rest
   static bool first_time = true;
   if (!first_time)
      return;
   first_time = false;

#ifndef NDEBUG
   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const int prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   size_t bytes_used = 0;
   bytes_used += sizeof(bool) * cached.global.size();
   bytes_used += sizeof(bool) * cached.local.size();
   bytes_used += sizeof(real) * alpha.size();
   bytes_used += sizeof(real) * beta.size();
   bytes_used += sizeof(real) * gamma.global.size();
   bytes_used += sizeof(real) * gamma.local.size();
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.flags(flags);
#endif

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "mn_max = " << mn_max << std::endl;
   std::cerr << "mn_min = " << mn_min << std::endl;
   std::cerr << "alpha = " << N + 1 << "×" << mtau_max - mtau_min + 1 << " = " << alpha.size() << std::endl;
   std::cerr << "beta = " << N + 1 << "×" << mtau_max - mtau_min + 1 << " = " << beta.size() << std::endl;
   if (globalstore)
      {
      std::cerr << "gamma = " << q << "×" << N << "×" << mtau_max - mtau_min + 1 << "×" << mn_max - mn_min + 1 << " = " << gamma.global.size() << std::endl;
      if (lazy)
         std::cerr << "cached = " << N << "×" << mtau_max - mtau_min + 1 << " = " << cached.global.size() << std::endl;
      }
   else
      {
      std::cerr << "gamma = " << q << "×" << mtau_max - mtau_min + 1 << "×" << mn_max - mn_min + 1 << " = " << gamma.local.size() << std::endl;
      if (lazy)
         std::cerr << "cached = " << computer.depth << "×" << mtau_max - mtau_min + 1 << " = " << cached.local.size() << std::endl;
      }
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::free()
   {
   alpha.init(0, 0);
   beta.init(0, 0);
   gamma.global.init(0, 0);
   gamma.local.init(0, 0);
   cached.global.init(0, 0);
   cached.local.init(0, 0);
   // copy over to references
   computer.alpha = alpha;
   computer.beta = beta;
   computer.gamma.global = gamma.global;
   computer.gamma.local = gamma.local;
   computer.cached.global = cached.global;
   computer.cached.local = cached.local;
   // flag the state of the arrays
   initialised = false;
   }

// helper methods

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::reset_cache() const
   {
   // initialise array and cache flags
   if (globalstore)
      {
      gamma.global.fill(0);
      cached.global.fill(0);
      }
   else
      {
      gamma.local.fill(0);
      cached.local.fill(0);
      }
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::print_gamma(std::ostream& sout) const
   {
   // copy the data set from the device
   array2r_t host_gamma = array2r_t(gamma.global);
   sout << "gamma = " << std::endl;
   for (int i = 0; i < computer.N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int d = 0; d < computer.q; d++)
         {
         sout << "d = " << d << ":" << std::endl;
         for (int x = -computer.mtau_max; x <= computer.mtau_max; x++)
            {
            for (int deltax = computer.mn_min; deltax <= computer.mn_max; deltax++)
               {
               const int ndx = computer.get_gamma_index(d, x, deltax);
               sout << '\t' << host_gamma(i, ndx);
               }
            sout << std::endl;
            }
         }
      }
   }

// data movement

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::copy_table(const dev_array2r_t& dev_table,
      array1vr_t& table)
   {
   // determine source sizes
   const int rows = dev_table.get_rows();
   const int cols = dev_table.get_cols();
   // copy from device in a single operation
   array1r_t data = array1r_t(dev_table);
   // initialise result table and copy one row at a time
   libbase::allocate(table, rows, cols);
   for (int i = 0; i < rows; i++)
      table(i) = data.extract(i * cols, cols);
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::copy_table(const array1vd_t& table,
      dev_array2r_t& dev_table)
   {
   // determine source sizes
   const int rows = table.size();
   const int cols = (rows > 0) ? table(0).size() : 0;
   // initialise contiguous vector and copy one row at a time
   array1r_t data(rows * cols);
   for (int i = 0; i < rows; i++)
      {
      assert(table(i).size() == cols);
      data.segment(i * cols, cols) = table(i);
      }
   // initialize result table and copy to device in a single operation
   dev_table.init(rows, cols);
   dev_table = data;
   }

//! Set up device with copies of the necessary data
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::setup(libcomm::instrumented& collector,
      const array1s_t& r, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, const int offset)
   {
   assert( initialised);
   // start timer
   libbase::cputimer ts("t_setup");
   // Validate sizes and offset
   assertalways(offset == -computer.mtau_min);
   assertalways(r.size() == computer.tau + computer.mtau_max - computer.mtau_min);
   assertalways(sof_prior.size() == computer.mtau_max - computer.mtau_min + 1);
   assertalways(eof_prior.size() == computer.mtau_max - computer.mtau_min + 1);
   // copy input data to device, allocating space as needed
   dev_r = r;
   copy_table(app, dev_app);
   dev_sof_table = array1r_t(sof_prior);
   dev_eof_table = array1r_t(eof_prior);
   // allocate space on device for result
   dev_ptable.init(computer.N, computer.q);
   // create a copy of the device object (to pass to kernels)
   dev_object = computer;
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(ts);
#if DEBUG>=3
   // show input data, as on device
   std::cerr << "r = " << array1s_t(dev_r) << std::endl;
   std::cerr << "app = " << array2r_t(dev_app) << std::endl;
   std::cerr << "sof_prior = " << array1r_t(dev_sof_table) << std::endl;
   std::cerr << "eof_prior = " << array1r_t(dev_eof_table) << std::endl;
#endif
   }

//! Prepare copies of input data for lazy/local gamma computations

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::setup_gamma(
      libcomm::instrumented& collector)
   {
   // start timer
   libbase::cputimer tg("t_gamma_setup");
   // keep a copy of received vector and a-priori statistics
   // (we need them later when computing gamma lazily or locally)
   computer.r = dev_r;
   computer.app = dev_app;
   // re-create a copy of the device object (to pass to kernels)
   dev_object = computer;
   // reset cache values if necessary
   if (lazy)
      reset_cache();
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(tg);
   }

//! Results transfer

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::transfer(
      libcomm::instrumented& collector, array1vr_t& ptable, array1r_t& sof_post,
      array1r_t& eof_post)
   {
   libbase::cputimer tc("t_transfer");
   copy_table(dev_ptable, ptable);
   sof_post = array1r_t(dev_sof_table);
   eof_post = array1r_t(dev_eof_table);
   collector.add_timer(tc);
   }

// decode functions - global path (implemented using kernel calls)

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::work_gamma(
      libcomm::instrumented& collector, const dev_array1s_t& r,
      const dev_array2r_t& app)
   {
   assert( initialised);
   // start timer
   libbase::cputimer tg("t_gamma");
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   const int mn_max = computer.mn_max;
   const int sm_all = cudaGetMultiprocessorCount();
   // set up kernel sizes
   // Gamma computation:
   // block index is for (i,x) where:
   //   i in [0, N-1]: x-grid size = N
   //   x in [mtau_min, mtau_max]: y-grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real2 array of size (n + mn_max + 1) for each thread
   const int gamma_thread_x = std::min(q, gamma_thread_count);
   const int gamma_thread_y = std::min((N * Mtau + sm_all - 1) / sm_all,
         std::min(Mtau, gamma_thread_count / gamma_thread_x));
   const dim3 gridDimG(N, (Mtau + gamma_thread_y - 1) / gamma_thread_y);
   const dim3 blockDimG(gamma_thread_x, gamma_thread_y);
   const size_t sharedMemG = computer.receiver.receiver_sharedmem() * count(blockDimG);
   // store kernel parameters
   add_kernel_parameters(collector, "gamma", gridDimG, blockDimG, sharedMemG,
         fba2_global_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // global pre-computation of gamma values
   fba2_global_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimG,blockDimG,sharedMemG>>>(dev_object, r, app);
   cudaSafeCall(cudaPeekAtLastError());
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(tg);
#if DEBUG>=3
   print_gamma(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::work_alpha_and_beta(
      libcomm::instrumented& collector, const dev_array1r_t& sof_prior,
      const dev_array1r_t& eof_prior)
   {
   assert( initialised);
   // start timer
   libbase::cputimer tab("t_alpha+beta");
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   const int sm_half = cudaGetMultiprocessorCount() / 2;
   // set up kernel sizes
   // Alpha (and Beta) initialization:
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimI(1);
   const dim3 blockDimI(Mtau);
   // Alpha computation:
   // block index is for x2 in [mtau_min, mtau_max]: grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real arrays of size q, Mtau for every group of q threads
   const int alpha_thread_x = std::min(q, alpha_thread_count);
   const int alpha_thread_y = std::min((Mtau + sm_half - 1) / sm_half,
         alpha_thread_count / alpha_thread_x);
   const dim3 gridDimA((Mtau + alpha_thread_y - 1) / alpha_thread_y);
   const dim3 blockDimA(alpha_thread_x, alpha_thread_y);
   const size_t sharedMemA = (q + Mtau) * alpha_thread_y * sizeof(real);
   // Beta computation:
   // block index is for x2 in [mtau_min, mtau_max]: grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real arrays of size q, Mtau for every group of q threads
   const int beta_thread_x = std::min(q, beta_thread_count);
   const int beta_thread_y = std::min((Mtau + sm_half - 1) / sm_half,
         beta_thread_count / beta_thread_x);
   const dim3 gridDimB((Mtau + beta_thread_y - 1) / beta_thread_y);
   const dim3 blockDimB(beta_thread_x, beta_thread_y);
   const size_t sharedMemB = (q + Mtau) * beta_thread_y * sizeof(real);
   // Normalization computation:
   // NOTE: this has to be done in one block, as we need to sync after
   //       determining the scale to use
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimN(1);
   const dim3 blockDimN(Mtau);
   // store kernel parameters
   add_kernel_parameters(collector, "alpha_init", gridDimI, blockDimI, 0,
         fba2_init_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta_init", gridDimI, blockDimI, 0,
         fba2_init_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "alpha", gridDimA, blockDimA, sharedMemA,
         fba2_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta", gridDimB, blockDimB, sharedMemB,
         fba2_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "alpha_norm", gridDimN, blockDimN, 0,
         fba2_normalize_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta_norm", gridDimN, blockDimN, 0,
         fba2_normalize_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Set up streams to parallelize alpha/beta computations
   stream sa, sb;
   // Alpha + Beta initialization:
   fba2_init_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimI,blockDimI,0,sa.get_id()>>>(dev_object, sof_prior);
   cudaSafeCall(cudaPeekAtLastError());
   fba2_init_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimI,blockDimI,0,sb.get_id()>>>(dev_object, eof_prior);
   cudaSafeCall(cudaPeekAtLastError());
   // normalize
   fba2_normalize_alpha_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,sa.get_id()>>>(dev_object, 0);
   cudaSafeCall(cudaPeekAtLastError());
   fba2_normalize_beta_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,sb.get_id()>>>(dev_object, N);
   cudaSafeCall(cudaPeekAtLastError());
   // Alpha + Beta computations:
   // Alpha computes from 1 to N, inclusive (after init at 0)
   // Beta starts from N-1 to 0, inclusive (after init at N)
   for (int i = 1; i <= N; i++)
      {
      fba2_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimA,blockDimA,sharedMemA,sa.get_id()>>>(dev_object, i);
      cudaSafeCall(cudaPeekAtLastError());
      fba2_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimB,blockDimB,sharedMemB,sb.get_id()>>>(dev_object, N - i);
      cudaSafeCall(cudaPeekAtLastError());
      // normalize
      fba2_normalize_alpha_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,sa.get_id()>>>(dev_object, i);
      cudaSafeCall(cudaPeekAtLastError());
      fba2_normalize_beta_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,sb.get_id()>>>(dev_object, N - i);
      cudaSafeCall(cudaPeekAtLastError());
      }
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(tab);
#if DEBUG>=3
   std::cerr << "alpha = " << libbase::matrix<real>(alpha) << std::endl;
   std::cerr << "beta = " << libbase::matrix<real>(beta) << std::endl;
   // show gamma as well if computing lazily
   if (lazy)
      print_gamma(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::work_results(
      libcomm::instrumented& collector, dev_array2r_t& ptable,
      dev_array1r_t& sof_post, dev_array1r_t& eof_post) const
   {
   assert( initialised);
   // start timer
   libbase::cputimer tr("t_results");
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   // set up kernel sizes
   // compute APPs of message
   // block index is for i in [0, N-1]: grid size = N
   // thread index is for d in [0, q-1]: block size = q
   // shared memory: two real arrays of size Mtau
   const dim3 gridDimR(N, q / message_thread_count);
   const dim3 blockDimR(message_thread_count);
   const size_t sharedMemR = 2 * (Mtau) * sizeof(real);
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimS(1);
   const dim3 blockDimS(Mtau);
   // store kernel parameters
   add_kernel_parameters(collector, "message", gridDimR, blockDimR, sharedMemR,
         fba2_global_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "state", gridDimS, blockDimS, 0,
         fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Results computation:
   // compute APPs of message
   fba2_global_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimR,blockDimR,sharedMemR>>>(dev_object, ptable);
   cudaSafeCall(cudaPeekAtLastError());
   // compute APPs of sof/eof state values
   fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, sof_post, 0);
   cudaSafeCall(cudaPeekAtLastError());
   fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, eof_post, N);
   cudaSafeCall(cudaPeekAtLastError());
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(tr);
#if DEBUG>=3
   array1vr_t ptable;
   copy_table(dev_ptable, ptable);
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << array1r_t(dev_sof_table) << std::endl;
   std::cerr << "eof_post = " << array1r_t(dev_eof_table) << std::endl;
#endif
   }

// decode functions - local path (implemented using kernel calls)

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::work_alpha(
      libcomm::instrumented& collector, const dev_array1r_t& sof_prior)
   {
   assert( initialised);
   // start timer
   libbase::cputimer ta("t_alpha");
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   const int mn_max = computer.mn_max;
   const int sm_all = cudaGetMultiprocessorCount();
   const int sm_part = cudaGetMultiprocessorCount() / computer.depth;
   // set up kernel sizes
   // Gamma computation:
   // block index is for (i,x) where:
   //   i is not used: x-grid size = 1
   //   x in [mtau_min, mtau_max]: y-grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real2 array of size (n + mn_max + 1) for each thread
   const int gamma_thread_x = std::min(q, gamma_thread_count);
   const int gamma_thread_y = std::min((Mtau + sm_part - 1) / sm_part,
         gamma_thread_count / gamma_thread_x);
   const dim3 gridDimG(1, (Mtau + gamma_thread_y - 1) / gamma_thread_y);
   const dim3 blockDimG(gamma_thread_x, gamma_thread_y);
   const size_t sharedMemG = computer.receiver.receiver_sharedmem() * count(blockDimG);
   // Alpha initialization:
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimI(1);
   const dim3 blockDimI(Mtau);
   // Alpha computation:
   // block index is for x2 in [mtau_min, mtau_max]: grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real arrays of size q, Mtau for every group of q threads
   const int alpha_thread_x = std::min(q, alpha_thread_count);
   const int alpha_thread_y = std::min((Mtau + sm_all - 1) / sm_all,
         alpha_thread_count / alpha_thread_x);
   const dim3 gridDimA((Mtau + alpha_thread_y - 1) / alpha_thread_y);
   const dim3 blockDimA(alpha_thread_x, alpha_thread_y);
   const size_t sharedMemA = (q + Mtau) * alpha_thread_y * sizeof(real);
   // Normalization computation:
   // NOTE: this has to be done in one block, as we need to sync after
   //       determining the scale to use
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimN(1);
   const dim3 blockDimN(Mtau);
   // store kernel parameters
   add_kernel_parameters(collector, "gamma", gridDimG, blockDimG, sharedMemG,
         fba2_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "alpha_init", gridDimI, blockDimI, 0,
         fba2_init_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "alpha", gridDimA, blockDimA, sharedMemA,
         fba2_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "alpha_norm", gridDimN, blockDimN, 0,
         fba2_normalize_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Set up streams and events to pipeline gamma+alpha+norm computations
   stream s[computer.depth];
   event e[computer.depth];
   // Alpha initialization:
   {
   // shorthand
   const int i = 0;
   const int bi = i & (computer.depth-1);
   fba2_init_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimI,blockDimI,0,s[bi].get_id()>>>(dev_object, sof_prior);
   cudaSafeCall(cudaPeekAtLastError());
   // normalize
   fba2_normalize_alpha_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,s[bi].get_id()>>>(dev_object, i);
   cudaSafeCall(cudaPeekAtLastError());
   // event to sync dependent alpha computation
   e[bi].record(s[bi]);
   }
   // Alpha computation:
   for (int i_base = 1; i_base <= N; i_base += computer.depth)
      {
      // breadth-first issue of gamma computation
      for (int i = i_base; i < i_base + computer.depth && i <= N; i++)
         {
         // shorthand
         const int bi = i & (computer.depth-1);
         // pre-compute local gamma values, if necessary
         if (!lazy)
            {
            fba2_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimG,blockDimG,sharedMemG,s[bi].get_id()>>>(dev_object, dev_r, dev_app, i - 1);
            cudaSafeCall(cudaPeekAtLastError());
            }
         // reset local cache, if necessary
         else
            {
            // TODO: make async on stream
            gamma.local.extract_row(bi).fill(0);
            cached.local.extract_row(bi).fill(0);
            }
         }
      // breadth-first issue of alpha computation and normalization
      for (int i = i_base; i < i_base + computer.depth && i <= N; i++)
         {
         // shorthand
         const int bi = i & (computer.depth-1);
         // alpha(i) depends on alpha(i-1)
         s[bi].wait(e[(i-1) & (computer.depth-1)]);
         fba2_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimA,blockDimA,sharedMemA,s[bi].get_id()>>>(dev_object, i);
         cudaSafeCall(cudaPeekAtLastError());
         // normalize
         fba2_normalize_alpha_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,s[bi].get_id()>>>(dev_object, i);
         cudaSafeCall(cudaPeekAtLastError());
         // event to sync dependent alpha computation
         e[bi].record(s[bi]);
         }
      }
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(ta);
#if DEBUG>=3
   std::cerr << "alpha = " << libbase::matrix<real>(alpha) << std::endl;
#endif
   }

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::work_beta_and_results(
      libcomm::instrumented& collector, const dev_array1r_t& eof_prior,
      dev_array2r_t& ptable, dev_array1r_t& sof_post, dev_array1r_t& eof_post)
   {
   assert( initialised);
   // start timer
   libbase::cputimer tbr("t_beta+results");
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   const int mn_max = computer.mn_max;
   const int sm_all = cudaGetMultiprocessorCount();
   const int sm_part = cudaGetMultiprocessorCount() / computer.depth;
   // set up kernel sizes
   // Gamma computation:
   // block index is for (i,x) where:
   //   i is not used: x-grid size = 1
   //   x in [mtau_min, mtau_max]: y-grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real2 array of size (n + mn_max + 1) for each thread
   const int gamma_thread_x = std::min(q, gamma_thread_count);
   const int gamma_thread_y = std::min((Mtau + sm_part - 1) / sm_part,
         gamma_thread_count / gamma_thread_x);
   const dim3 gridDimG(1, (Mtau + gamma_thread_y - 1) / gamma_thread_y);
   const dim3 blockDimG(gamma_thread_x, gamma_thread_y);
   const size_t sharedMemG = computer.receiver.receiver_sharedmem() * count(blockDimG);
   // Beta initialization:
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimI(1);
   const dim3 blockDimI(Mtau);
   // Beta computation:
   // block index is for x2 in [mtau_min, mtau_max]: grid size = Mtau
   //      (could be less if y-block size > 1)
   // thread index is for (d,xx), where:
   //   d in [0, q-1]: x-block size = q
   //      (could be less if restricted by device; compensated by loop)
   //   xx in [1, ..]: y-block size is such to maximise occupancy
   // shared memory: real arrays of size q, Mtau for every group of q threads
   const int beta_thread_x = std::min(q, beta_thread_count);
   const int beta_thread_y = std::min((Mtau + sm_all - 1) / sm_all,
         beta_thread_count / beta_thread_x);
   const dim3 gridDimB((Mtau + beta_thread_y - 1) / beta_thread_y);
   const dim3 blockDimB(beta_thread_x, beta_thread_y);
   const size_t sharedMemB = (q + Mtau) * beta_thread_y * sizeof(real);
   // Normalization computation:
   // NOTE: this has to be done in one block, as we need to sync after
   //       determining the scale to use
   // block index is not used: grid size = 1
   // thread index is for x2 in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimN(1);
   const dim3 blockDimN(Mtau);
   // compute APPs of message
   // block index is not used: grid size = 1
   // thread index is for d in [0, q-1]: block size = q
   // shared memory: two real arrays of size Mtau
   const dim3 gridDimR(1, q / message_thread_count);
   const dim3 blockDimR(message_thread_count);
   const size_t sharedMemR = 2 * Mtau * sizeof(real);
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimS(1);
   const dim3 blockDimS(Mtau);
   // store kernel parameters
   add_kernel_parameters(collector, "gamma", gridDimG, blockDimG, sharedMemG,
         fba2_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta_init", gridDimI, blockDimI, 0,
         fba2_init_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta", gridDimB, blockDimB, sharedMemB,
         fba2_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "beta_norm", gridDimN, blockDimN, 0,
         fba2_normalize_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "message", gridDimR, blockDimR, sharedMemR,
         fba2_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   add_kernel_parameters(collector, "state", gridDimS, blockDimS, 0,
         fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Set up streams and events to pipeline gamma+beta+norm+result computations
   stream s[computer.depth];
   event e[computer.depth];
   // Beta initialization:
   {
   // shorthand
   const int i = N;
   const int bi = i & (computer.depth-1);
   fba2_init_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimI,blockDimI,0,s[bi].get_id()>>>(dev_object, eof_prior);
   cudaSafeCall(cudaPeekAtLastError());
   // normalize
   fba2_normalize_beta_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,s[bi].get_id()>>>(dev_object, i);
   cudaSafeCall(cudaPeekAtLastError());
   // event to sync dependent beta computation
   e[bi].record(s[bi]);
   }
   // Beta + Results computation:
   for (int i_base = N-1; i_base >= 0; i_base -= computer.depth)
      {
      // breadth-first issue of gamma computation
      for (int i = i_base; i > i_base - computer.depth && i >= 0; i--)
         {
         // shorthand
         const int bi = i & (computer.depth-1);
         // pre-compute local gamma values, if necessary
         if (!lazy)
            {
            fba2_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimG,blockDimG,sharedMemG,s[bi].get_id()>>>(dev_object, dev_r, dev_app, i);
            cudaSafeCall(cudaPeekAtLastError());
            }
         // reset local cache, if necessary
         else
            {
            // TODO: make async on stream
            gamma.local.extract_row(bi).fill(0);
            cached.local.extract_row(bi).fill(0);
            }
         }
      // breadth-first issue of beta computation and normalization
      for (int i = i_base; i > i_base - computer.depth && i >= 0; i--)
         {
         // shorthand
         const int bi = i & (computer.depth-1);
         // beta(i) depends on beta(i+1)
         s[bi].wait(e[(i+1) & (computer.depth-1)]);
         fba2_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimB,blockDimB,sharedMemB,s[bi].get_id()>>>(dev_object, i);
         cudaSafeCall(cudaPeekAtLastError());
         // normalize
         fba2_normalize_beta_kernel <receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimN,blockDimN,0,s[bi].get_id()>>>(dev_object, i);
         cudaSafeCall(cudaPeekAtLastError());
         // event to sync dependent beta computation
         e[bi].record(s[bi]);
         // compute partial result
         fba2_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimR,blockDimR,sharedMemR,s[bi].get_id()>>>(dev_object, ptable, i);
         cudaSafeCall(cudaPeekAtLastError());
         }
      }
   // compute APPs of sof/eof state values
   fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, sof_post, 0);
   cudaSafeCall(cudaPeekAtLastError());
   fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, eof_post, N);
   cudaSafeCall(cudaPeekAtLastError());
   // store timer
   cudaSafeDeviceSynchronize();
   collector.add_timer(tbr);
#if DEBUG>=3
   std::cerr << "beta = " << libbase::matrix<real>(beta) << std::endl;
   array1vr_t ptable;
   copy_table(dev_ptable, ptable);
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << array1r_t(dev_sof_table) << std::endl;
   std::cerr << "eof_post = " << array1r_t(dev_eof_table) << std::endl;
#endif
   }

// User procedures

// Initialization

template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::init(
      int N, int q, int mtau_min, int mtau_max, int mn_min, int mn_max,
      int m1_min, int m1_max, double th_inner, double th_outer,
      const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer)
   {
   // Initialize our embedded metric computer with unchanging elements
   // (needs to happen before fba initialization)
   this->computer.receiver.init(computer);
   // if any parameters that effect memory have changed, release memory
   if (initialised
         && (N != this->computer.N || q != this->computer.q
               || mtau_min != this->computer.mtau_min
               || mtau_max != this->computer.mtau_max
               || mn_min != this->computer.mn_min
               || mn_max != this->computer.mn_max))
      {
      free();
      }
   // code parameters
   assert(N > 0);
   this->computer.N = N;
   assert(q > 1);
   this->computer.q = q;
   // decoder parameters
   assert(mtau_min <= 0);
   assert(mtau_max >= 0);
   this->computer.mtau_min = mtau_min;
   this->computer.mtau_max = mtau_max;
   assert(mn_min <= 0);
   assert(mn_max >= 0);
   this->computer.mn_min = mn_min;
   this->computer.mn_max = mn_max;
   assert(m1_min <= 0);
   assert(m1_max >= 0);
   this->computer.m1_min = m1_min;
   this->computer.m1_max = m1_max;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   assert(thresholding || (th_inner == 0 && th_outer == 0));
   this->computer.th_inner = th_inner;
   this->computer.th_outer = th_outer;
   // determine thread count to use for gamma,alpha,beta,message kernels
   const int Mtau = mtau_max - mtau_min + 1;
   const size_t gamma_smem = this->computer.receiver.receiver_sharedmem();
   if (globalstore)
      gamma_thread_count = determine_thread_count(q * Mtau, 0, gamma_smem, 1,
            fba2_global_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   else
      gamma_thread_count = determine_thread_count(q * Mtau, 0, gamma_smem, 1,
            fba2_gamma_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   alpha_thread_count = determine_thread_count(q * Mtau, 0, (q + Mtau) * sizeof(real), q,
         fba2_alpha_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   beta_thread_count = determine_thread_count(q * Mtau, 0, (q + Mtau) * sizeof(real), q,
         fba2_beta_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   if (globalstore)
      message_thread_count = determine_thread_count(q, 2 * Mtau * sizeof(real), 0, 1,
            fba2_global_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   else
      message_thread_count = determine_thread_count(q, 2 * Mtau * sizeof(real), 0, 1,
            fba2_message_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   }

/*!
 * \brief Frame decode cycle
 * \param[in] collector Reference to (instrumented) results collector object
 * \param[in] r Received frame
 * \param[in] sof_prior Prior probabilities for start-of-frame position
 *                      (zero-index matches zero-index of r)
 * \param[in] eof_prior Prior probabilities for end-of-frame position
 *                      (zero-index matches tau-index of r, where tau is the
 *                      length of the transmitted frame)
 * \param[in] app A-Priori Probabilities for message
 * \param[out] ptable Posterior Probabilities for message
 * \param[out] sof_post Posterior probabilities for start-of-frame position
 *                      (indexing same as prior)
 * \param[out] eof_post Posterior probabilities for end-of-frame position
 *                      (indexing same as prior)
 * \param[in] offset Index offset for prior, post, and r vectors
 *
 * \note If APP table is empty, it is assumed that symbols are equiprobable.
 *
 * \note Priors for start and end-of-frame *must* be supplied; in the case of a
 *       received frame with exactly known boundaries, this must be offset by
 *       mtau_max and padded to a total length of tau + mtau_max-mtau_min, where tau is the
 *       length of the transmitted frame. This avoids special handling for such
 *       vectors.
 *
 * \note Offset is the same as for stream_modulator.
 */
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::decode(libcomm::instrumented& collector,
      const array1s_t& r, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post, const int offset)
   {
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << computer.N << std::endl;
   std::cerr << "q = " << computer.q << std::endl;
   std::cerr << "m1_min = " << computer.m1_min << std::endl;
   std::cerr << "m1_max = " << computer.m1_max << std::endl;
   std::cerr << "mtau_min = " << computer.mtau_min << std::endl;
   std::cerr << "mtau_max = " << computer.mtau_max << std::endl;
   std::cerr << "mn_min = " << mn_min << std::endl;
   std::cerr << "mn_max = " << mn_max << std::endl;
   std::cerr << "th_inner = " << computer.th_inner << std::endl;
   std::cerr << "th_outer = " << computer.th_outer << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   // Initialise memory on device if necessary
   if (!initialised)
      allocate();
   // Setup device
   setup(collector, r, sof_prior, eof_prior, app, offset);
   // Gamma
   if (!lazy && globalstore)
      {
      // compute immediately for global pre-compute mode
      work_gamma(collector, dev_r, dev_app);
      }
   else
      {
      // prepare copies of input data for lazy/local gamma computations
      setup_gamma(collector);
      }
   // Alpha + Beta + Results
   if (globalstore)
      {
      // Alpha + Beta
      work_alpha_and_beta(collector, dev_sof_table, dev_eof_table);
      // Compute results
      work_results(collector, dev_ptable, dev_sof_table, dev_eof_table);
      }
   else
      {
      // Alpha
      work_alpha(collector, dev_sof_table);
      // Beta
      work_beta_and_results(collector, dev_eof_table, dev_ptable, dev_sof_table, dev_eof_table);
      }
   // Results transfer
   transfer(collector, ptable, sof_post, eof_post);

   // add values for limits that depend on channel conditions
   collector.add_timer(computer.mtau_min, "c_mtau_min");
   collector.add_timer(computer.mtau_max, "c_mtau_max");
   collector.add_timer(computer.mn_min, "c_mn_min");
   collector.add_timer(computer.mn_max, "c_mn_max");
   collector.add_timer(computer.m1_min, "c_m1_min");
   collector.add_timer(computer.m1_max, "c_m1_max");
   // add memory usage
   collector.add_timer(sizeof(real) * alpha.size(), "m_alpha");
   collector.add_timer(sizeof(real) * beta.size(), "m_beta");
   collector.add_timer(sizeof(real) * (gamma.global.size() + gamma.local.size()), "m_gamma");
   }

/*!
 * \brief Get the posterior channel drift pdf at specified boundary
 * \param[out] pdf Posterior Probabilities for codeword boundary
 *
 * Codeword boundaries are taken to include frame boundaries, such that
 * index 'i' corresponds to the boundary between codewords 'i' and 'i+1'.
 * This method must be called after a call to decode(), so that it can return
 * posteriors for the last transmitted frame.
 */
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::get_drift_pdf(array1r_t& pdf, const int i) const
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   // set up kernel sizes
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimS(1);
   const dim3 blockDimS(Mtau);
   // store kernel parameters
   //add_kernel_parameters(collector, "state", gridDimS, blockDimS, 0,
   //      fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Drift PDF computation:
   assert(i >= 0 && i <= N);
   fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, dev_sof_table, i);
   cudaSafeCall(cudaPeekAtLastError());
   // copy result from temporary space
   pdf = array1r_t(dev_sof_table);
   }

/*!
 * \brief Get the posterior channel drift pdf at codeword boundaries
 * \param[out] pdftable Posterior Probabilities for codeword boundaries
 *
 * Codeword boundaries are taken to include frame boundaries, such that
 * pdftable(i) corresponds to the boundary between codewords 'i' and 'i+1'.
 * This method must be called after a call to decode(), so that it can return
 * posteriors for the last transmitted frame.
 */
template <class receiver_t, class sig, class real, class real2, bool thresholding, bool lazy, bool globalstore>
void fba2<receiver_t, sig, real, real2, thresholding, lazy, globalstore>::get_drift_pdf(array1vr_t& pdftable) const
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int Mtau = computer.mtau_max - computer.mtau_min + 1;
   // set up kernel sizes
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [mtau_min, mtau_max]: block size = Mtau
   const dim3 gridDimS(1);
   const dim3 blockDimS(Mtau);
   // store kernel parameters
   //add_kernel_parameters(collector, "state", gridDimS, blockDimS, 0,
   //      fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore>);
   // Drift PDF computation:
   // allocate space for results
   pdftable.init(N + 1);
   // consider each time index in the order given
   for (int i = 0; i <= N; i++)
      {
      fba2_state_app_kernel<receiver_t, sig, real, real2, thresholding, lazy, globalstore> <<<gridDimS,blockDimS,0>>>(dev_object, dev_sof_table, i);
      cudaSafeCall(cudaPeekAtLastError());
      // copy result from temporary space
      pdftable(i) = array1r_t(dev_sof_table);
      }
   }

} // end namespace

/* \note There are no explicit realizations here, as for this module we need to
 * split the realizations over separate units, or ptxas will complain with
 * excessive cmem usage. All realizations are in the fba2-cuda-instXX.cu files.
 */
