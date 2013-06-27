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

#include "fba2-cuda.h"
#include "pacifier.h"
#include "vectorutils.h"
#include "cuda/gputimer.h"
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

// common small tasks

template <class receiver_t, class sig, class real, class real2>
__device__
real fba2<receiver_t, sig, real, real2>::metric_computer::get_threshold(const dev_array2r_ref_t& metric, int row, int cols, real factor)
   {
   const bool thresholding = (factor > 0);
   real threshold = 0;
   if (thresholding)
      {
      for (int col = 0; col < cols; col++)
         {
         if (metric(row, col) > threshold)
            {
            threshold = metric(row, col);
            }
         }
      threshold *= factor;
      }
   return threshold;
   }

/*! \brief Returns the sum of the elements in the given array (length N)
 * The sum is computed in parallel between the threads in a given block.
 * A limitation for this to work is that N must be a multiple of 2, and
 * the block size has to be at least N/2 threads.
 * \warning The contents of the array are destroyed in the process.
 */
template <class receiver_t, class sig, class real, class real2>
__device__
real fba2<receiver_t, sig, real, real2>::metric_computer::parallel_sum(real array[], const int N)
   {
   const int i = threadIdx.x;
   cuda_assert(N % 2 == 0);
   cuda_assert(N / 2 <= blockDim.x); // Total number of active threads
   for(int n = N; n > 1; n >>= 1)
      {
      const int half = (n >> 1); // divide by two
      // only the first half of the threads will be active.
      if (i < half)
         {
         array[i] += array[i + half];
         }
      // wait until all threads have completed their part
      __syncthreads();
      }
   return array[0];
   }

template <class receiver_t, class sig, class real, class real2>
__device__
real fba2<receiver_t, sig, real, real2>::metric_computer::get_scale(const dev_array2r_ref_t& metric, int row, int cols)
   {
   real scale = 0;
   for (int col = 0; col < cols; col++)
      {
      scale += metric(row, col);
      }
   cuda_assertalways(scale > real(0));
   scale = real(1) / scale;
   return scale;
   }

template <class receiver_t, class sig, class real, class real2>
__device__
void fba2<receiver_t, sig, real, real2>::metric_computer::normalize(dev_array2r_ref_t& metric, int row, int cols)
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

// decode functions - partial computations

template <class receiver_t, class sig, class real, class real2>
__device__
void fba2<receiver_t, sig, real, real2>::metric_computer::work_alpha(const dev_array1r_ref_t& sof_prior, int i)
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_inner > 0);
   // get end drift from block index and symbol value from thread index
   const int x2 = blockIdx.x - xmax;
   const int d = threadIdx.x;
   // set up variables shared within block
   SharedMemory<real> shared;
   real* this_alpha = shared.getPointer();

   if(i == 0)
      {
      // set array initial conditions (parallelized):
      if (d == 0)
         {
         // set initial drift distribution
         alpha(0, x2 + xmax) = sof_prior(x2 + xmax);
         }
      }
   else
      {
      // compute remaining matrix values:
      // determine the strongest path at this point
      const real threshold = get_threshold(alpha, i - 1, 2 * xmax + 1, th_inner);
      // initialize result holder
      this_alpha[d] = 0;
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= dmax
      //   x2-x1 >= dmin
      const int x1min = max(-xmax, x2 - dmax);
      const int x1max = min(xmax, x2 - dmin);
      for (int x1 = x1min; x1 <= x1max; x1++)
         {
         // cache previous alpha value in a register
         const real prev_alpha = alpha(i - 1, x1 + xmax);
         // ignore paths below a certain threshold
         if (!thresholding || prev_alpha >= threshold)
            {
            // each block computes for a different end-state (x2)
            // each thread in a block is computing for a different symbol (d)
            real temp = prev_alpha;
            temp *= get_gamma(d, i - 1, x1, x2 - x1);
            this_alpha[d] += temp;
            }
         }
      // make sure all threads in block have finished updating this_alpha
      __syncthreads();
      // compute sum of shared array
      const real temp = parallel_sum(this_alpha, q);
      // store result (first thread in block)
      if (d == 0)
         {
         alpha(i, x2 + xmax) = temp;
         }
      }
   }

template <class receiver_t, class sig, class real, class real2>
__device__
void fba2<receiver_t, sig, real, real2>::metric_computer::work_beta(const dev_array1r_ref_t& eof_prior, int i)
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_inner > 0);
   // get start drift from block index and symbol value from thread index
   const int x1 = blockIdx.x - xmax;
   const int d = threadIdx.x;
   // set up variables shared within block
   SharedMemory<real> shared;
   real* this_beta = shared.getPointer();

   if(i == N)
      {
      // set array initial conditions (parallelized):
      if (d == 0)
         {
         // set final drift distribution
         beta(N, x1 + xmax) = eof_prior(x1 + xmax);
         }
      }
   else
      {
      // compute remaining matrix values:
      // determine the strongest path at this point
      const real threshold = get_threshold(beta, i + 1, 2 * xmax + 1, th_inner);
      // initialize result holder
      this_beta[d] = 0;
      // limits on deltax can be combined as (c.f. allocate() for details):
      //   x2-x1 <= dmax
      //   x2-x1 >= dmin
      const int x2min = max(-xmax, dmin + x1);
      const int x2max = min(xmax, dmax + x1);
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         // cache next beta value in a register
         const real next_beta = beta(i + 1, x2 + xmax);
         // ignore paths below a certain threshold
         if (!thresholding || next_beta >= threshold)
            {
            // each block computes for a different start-state (x1)
            // each thread in a block is computing for a different symbol (d)
            real temp = next_beta;
            temp *= get_gamma(d, i, x1, x2 - x1);
            this_beta[d] += temp;
            }
         }
      // make sure all threads in block have finished updating this_beta
      __syncthreads();
      // compute sum of shared array
      const real temp = parallel_sum(this_beta, q);
      // store result
      if (d == 0)
         {
         beta(i, x1 + xmax) = temp;
         }
      }
   }

template <class receiver_t, class sig, class real, class real2>
__device__
void fba2<receiver_t, sig, real, real2>::metric_computer::work_message_app(dev_array2r_ref_t& ptable, const int i) const
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_outer > 0);
   // Check result vector (one symbol per timestep)
   cuda_assertalways(ptable.get_rows()==N && ptable.get_cols()==q);
   // get symbol value from thread index
   const int d = threadIdx.x;
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   // - all threads are independent and indexes guaranteed in range
   // determine the strongest path at this point
   const real threshold = get_threshold(alpha, i, 2 * xmax + 1, th_outer);
   // initialize result holder
   real p = 0;
   for (int x1 = -xmax; x1 <= xmax; x1++)
      {
      // cache this alpha value in a register
      const real this_alpha = alpha(i, x1 + xmax);
      // ignore paths below a certain threshold
      if (!thresholding || this_alpha >= threshold)
         {
         // limits on deltax can be combined as (c.f. allocate() for details):
         //   x2-x1 <= dmax
         //   x2-x1 >= dmin
         const int x2min = max(-xmax, dmin + x1);
         const int x2max = min(xmax, dmax + x1);
         for (int x2 = x2min; x2 <= x2max; x2++)
            {
            real temp = this_alpha;
            temp *= beta(i + 1, x2 + xmax);
            temp *= get_gamma(d, i, x1, x2 - x1);
            p += temp;
            }
         }
      }
   // store result
   ptable(i,d) = p;
   }

template <class receiver_t, class sig, class real, class real2>
__device__
void fba2<receiver_t, sig, real, real2>::metric_computer::work_state_app(dev_array1r_ref_t& ptable,
      const int i) const
   {
   // Check result vector and requested index
   cuda_assertalways(ptable.size()==2*xmax+1);
   cuda_assert(i >= 0 && i <= N);
   // set up block & thread indexes
   const int x = threadIdx.x - xmax;
   // compute posterior probabilities for given index
   ptable(x + xmax) = alpha(i, x + xmax) * beta(i, x + xmax);
   }

// *** Kernels *** NOTE: all kernels *must* be global functions

// common small tasks

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_normalize_alpha_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const int i)
   {
   object().normalize_alpha(i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_normalize_beta_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const int i)
   {
   object().normalize_beta(i);
   }

// decode functions - partial computations

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_gamma_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const vector_reference<sig> r,
      const matrix_reference<real> app, const int i)
   {
   object().work_gamma(r, app, i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_alpha_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const vector_reference<real> sof_prior, const int i)
   {
   object().work_alpha(sof_prior, i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_beta_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const vector_reference<real> eof_prior, const int i)
   {
   object().work_beta(eof_prior, i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_message_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, matrix_reference<real> ptable, const int i)
   {
   object().work_message_app(ptable, i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_state_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, vector_reference<real> ptable, const int i)
   {
   object().work_state_app(ptable, i);
   }

// decode functions - global path

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_gamma_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, const vector_reference<sig> r,
      const matrix_reference<real> app)
   {
   // get symbol index from block index
   const int i = blockIdx.x;
   object().work_gamma(r, app, i);
   }

template <class receiver_t, class sig, class real, class real2>
__global__
void fba2_message_app_kernel(value_reference<typename fba2<receiver_t, sig, real, real2>::metric_computer> object, matrix_reference<real> ptable)
   {
   // get symbol index from block index
   const int i = blockIdx.x;
   object().work_message_app(ptable, i);
   }

// *** Main Class

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::allocate()
   {
   // flag the state of the arrays
   initialised = true;

   // determine allowed limits on deltax:
   // limits on insertions and deletions:
   //   x2-x1 <= n*I
   //   x2-x1 >= -n
   // limits on introduced drift in this section:
   // (necessary for forward recursion on extracted segment)
   //   x2-x1 <= dxmax
   //   x2-x1 >= -dxmax
   // the above two sets of limits can be combined as:
   //   x2-x1 <= min(n*I, dxmax) = dmax
   //   x2-x1 >= max(-n, -dxmax) = dmin
   computer.dmin = std::max(-computer.n, -computer.dxmax);
   computer.dmax = std::min(computer.n * computer.I, computer.dxmax);
   // alpha needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [0, N] and x in [-xmax, xmax]
   alpha.init(computer.N + 1, 2 * computer.xmax + 1); // offsets: 0, xmax
   beta.init(computer.N + 1, 2 * computer.xmax + 1); // offsets: 0, xmax

   if (computer.flags.globalstore)
      {
      /* gamma needs indices (i,x,d,deltax) where
       * i in [0, N-1]
       * x in [-xmax, xmax]
       * d in [0, q-1]
       * deltax in [dmin, dmax]
       * NOTE: this is allocated as a flat sequence
       */
      gamma.global.init(computer.N * (2 * computer.xmax + 1) * computer.q
                  * (computer.dmax - computer.dmin + 1));
      gamma.local.init(0);
      }
   else
      {
      /* gamma needs indices (x,d,deltax) where
       * x in [-xmax, xmax]
       * d in [0, q-1]
       * deltax in [dmin, dmax]
       * NOTE: this is allocated as a flat sequence
       */
      gamma.local.init((2 * computer.xmax + 1) * computer.q
                  * (computer.dmax - computer.dmin + 1));
      gamma.global.init(0);
      }
   // need to keep track only if we're caching lazy computations
   if (computer.flags.lazy)
      {
      if (computer.flags.globalstore)
         {
         /* cached needs indices (i,x) where
          * i in [0, N-1]
          * x in [-xmax, xmax]
          */
         cached.global.init(computer.N, 2 * computer.xmax + 1); // offsets: 0, xmax
         cached.local.init(0);
         }
      else
         {
         /* cached needs indices (x) where
          * x in [-xmax, xmax]
          */
         cached.local.init(2 * computer.xmax + 1); // offsets: xmax
         cached.global.init(0, 0);
         }
      }
   else
      {
      cached.global.init(0, 0);
      cached.local.init(0);
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

#ifndef NDEBUG
   // determine required space for inner metric table (Jiao-Armand method)
   size_t entries = 0;
   for (int delta = computer.dmin; delta <= computer.dmax; delta++)
      entries += (1 << (delta + computer.n));
   entries *= computer.q;
   std::cerr << "Jiao-Armand Table Size: "
         << sizeof(float) * entries / double(1 << 20) << "MiB" << std::endl;
#endif

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "dmax = " << computer.dmax << std::endl;
   std::cerr << "dmin = " << computer.dmin << std::endl;
   std::cerr << "alpha = " << computer.N + 1 << "x" << 2 * computer.xmax + 1
   << " = " << alpha.size() << std::endl;
   std::cerr << "beta = " << computer.N + 1 << "x" << 2 * computer.xmax + 1
   << " = " << beta.size() << std::endl;
   if (flags.globalstore)
      {
      std::cerr << "gamma = " << computer.q << "x" << computer.N << "x" << 2 * computer.xmax + 1 << "x"
      << computer.dmax - computer.dmin + 1 << " = " << gamma.global.size()
      << std::endl;
      if (flags.lazy)
         {
         std::cerr << "cached = " << computer.N << "x" << 2 * computer.xmax + 1 << " = "
         << cached.global.size() << std::endl;
         }
      }
   else
      {
      std::cerr << "gamma = " << computer.q << "x" << 2 * computer.xmax + 1 << "x" <<
            computer.dmax - computer.dmin + 1 << " = " << gamma.local.size() << std::endl;
      if (flags.lazy)
         {
         std::cerr << "cached = " << 2 * computer.xmax + 1 << " = "
         << cached.local.size() << std::endl;
         }
      }
#endif
   }

/*! \brief Release memory for working matrices
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::free()
   {
   alpha.init(0, 0);
   beta.init(0, 0);
   gamma.global.init(0);
   gamma.local.init(0);
   cached.global.init(0, 0);
   cached.local.init(0);
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

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::reset_cache() const
   {
   // initialise array and cache flags
   if (computer.flags.globalstore)
      {
      gamma.global.fill(real(0));
      cached.global.fill(false);
      }
   else
      {
      gamma.local.fill(real(0));
      cached.local.fill(false);
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::print_gamma(std::ostream& sout) const
   {
   // copy the data set from the device
   libbase::vector<real> host_gamma = libbase::vector<real>(gamma.global);
   sout << "gamma = " << std::endl;
   for (int i = 0; i < computer.N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int d = 0; d < computer.q; d++)
         {
         sout << "d = " << d << ":" << std::endl;
         for (int x = -computer.xmax; x <= computer.xmax; x++)
            {
            for (int deltax = computer.dmin; deltax <= computer.dmax; deltax++)
               {
               const int ndx = computer.get_gamma_index(d, i, x, deltax);
               sout << '\t' << host_gamma(ndx);
               }
            sout << std::endl;
            }
         }
      }
   }

// data movement

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::copy_table(const dev_array2r_t& dev_table,
      array1vr_t& table)
   {
   // determine source sizes
   const int rows = dev_table.get_rows();
   const int cols = dev_table.get_cols();
   // initialise result table and copy one row at a time
   libbase::allocate(table, rows, cols);
   for (int i = 0; i < rows; i++)
      {
      table(i) = array1r_t(dev_table.extract_row(i));
      }
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::copy_table(const array1vd_t& table,
      dev_array2r_t& dev_table)
   {
   // determine source sizes
   const int rows = table.size();
   const int cols = (rows > 0) ? table(0).size() : 0;
   // initialise result table and copy one row at a time
   dev_table.init(rows, cols);
   for (int i = 0; i < rows; i++)
      {
      assert(table(i).size() == cols);
      dev_table.extract_row(i) = array1r_t(table(i));
      }
   }

// decode functions - global path (implemented using kernel calls)

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_gamma(const dev_array1s_t& r,
      const dev_array2r_t& app)
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "Gamma Kernel: (" << N << "x" << 2 * xmax + 1 << ") x " << q << std::endl;
      first_time = false;
      }
   // global pre-computation of gamma values
   // block index is for (i,x) where:
   //   i in [0, N-1]: x-grid size = N
   //   x in [-xmax, xmax]: y-grid size = 2*xmax+1
   // thread index is for d in [0, q-1]: block size = q
   dim3 gridSize(N, 2*xmax+1);
   fba2_gamma_kernel<receiver_t, sig, real, real2> <<<gridSize,q>>>(dev_object, r, app);

#if DEBUG>=3
   print_gamma(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_alpha_and_beta(const dev_array1r_t& sof_prior,
      const dev_array1r_t& eof_prior)
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "Alpha Kernel: " << 2 * xmax + 1 << " x " << q << std::endl;
      std::cerr << "Beta Kernel: " << 2 * xmax + 1 << " x " << q << std::endl;
      if (computer.flags.norm)
         std::cerr << "Norm. Kernel: " << 1 << " x " << 2 * xmax + 1 << std::endl;
      first_time = false;
      }
   // Alpha + Beta computations:
   // Alpha starts from 0 to N, inclusive
   // Beta starts from N to 0, inclusive
   // TODO: run alpha/beta in separate streams to parallelize
   for (int i = 0; i <= N; i++)
      {
      // block index is for x2 in [-xmax, xmax]: grid size = 2*xmax+1
      // thread index is for d in [0, q-1]: block size = q
      // shared memory: array of q 'real's
      fba2_alpha_kernel<receiver_t, sig, real, real2> <<<2*xmax+1,q,q*sizeof(real)>>>(dev_object, sof_prior, i);
      fba2_beta_kernel<receiver_t, sig, real, real2> <<<2*xmax+1,q,q*sizeof(real)>>>(dev_object, eof_prior, N - i);
      // normalize if requested
      if (computer.flags.norm)
         {
         // NOTE: this has to be done in one block, as we need to sync after
         //       determining the scale to use 
         // block index is not used: grid size = 1
         // thread index is for x2 in [-xmax, xmax]: block size = 2*xmax+1
         fba2_normalize_alpha_kernel <receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, i);
         fba2_normalize_beta_kernel <receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, N - i);
         }
      }

#if DEBUG>=3
   std::cerr << "alpha = " << libbase::matrix<real>(alpha) << std::endl;
   std::cerr << "beta = " << libbase::matrix<real>(beta) << std::endl;
   // show gamma as well if computing lazily
   if (computer.flags.lazy)
      print_gamma(std::cerr);
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_results(dev_array2r_t& ptable,
      dev_array1r_t& sof_post, dev_array1r_t& eof_post) const
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "Message APP Kernel: " << N << " x " << q << std::endl;
      std::cerr << "State APP Kernel: " << 1 << " x " << 2 * xmax + 1 << std::endl;
      first_time = false;
      }
   // Results computation:
   // compute APPs of message
   // block index is for i in [0, N-1]: grid size = N
   // thread index is for d in [0, q-1]: block size = q
   fba2_message_app_kernel<receiver_t, sig, real, real2> <<<N,q>>>(dev_object, ptable);
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [-xmax, xmax]: block size = 2*xmax+1
   fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, sof_post, 0);
   fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, eof_post, N);

#if DEBUG>=3
   array1vr_t ptable;
   copy_table(dev_ptable, ptable);
   std::cerr << "ptable = " << ptable << std::endl;
   std::cerr << "sof_post = " << array1r_t(dev_sof_table) << std::endl;
   std::cerr << "eof_post = " << array1r_t(dev_eof_table) << std::endl;
#endif
   }

// decode functions - local path (implemented using kernel calls)

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_alpha(const dev_array1r_t& sof_prior)
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "Gamma Kernel: (1x" << 2 * xmax + 1 << ") x " << q << std::endl;
      std::cerr << "Alpha Kernel: " << 2 * xmax + 1 << " x " << q << std::endl;
      if (computer.flags.norm)
         std::cerr << "Norm. Kernel: " << 1 << " x " << 2 * xmax + 1 << std::endl;
      first_time = false;
      }
   // Alpha computation:
   for (int i = 0; i <= N; i++)
      {
      // local storage, index value where we need gamma values
      if (!computer.flags.globalstore && i > 0)
         {
         // pre-compute local gamma values, if necessary
         if (!computer.flags.lazy)
            {
            // global pre-computation of gamma values
            // block index is for (i,x) where:
            //   i is not used: x-grid size = 1
            //   x in [-xmax, xmax]: y-grid size = 2*xmax+1
            // thread index is for d in [0, q-1]: block size = q
            dim3 gridSize(1, 2*xmax+1);
            fba2_gamma_kernel<receiver_t, sig, real, real2> <<<gridSize,q>>>(dev_object, dev_r, dev_app, i - 1);
            }
         // reset local cache, if necessary
         else
            {
            gamma.local.fill(real(0));
            cached.local.fill(false);
            }
         }
      // block index is for x2 in [-xmax, xmax]: grid size = 2*xmax+1
      // thread index is for d in [0, q-1]: block size = q
      // shared memory: array of q 'real's
      fba2_alpha_kernel<receiver_t, sig, real, real2> <<<2*xmax+1,q,q*sizeof(real)>>>(dev_object, sof_prior, i);
      // normalize if requested
      if (computer.flags.norm)
         {
         // NOTE: this has to be done in one block, as we need to sync after
         //       determining the scale to use 
         // block index is not used: grid size = 1
         // thread index is for x2 in [-xmax, xmax]: block size = 2*xmax+1
         fba2_normalize_alpha_kernel <receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, i);
         }
      }

#if DEBUG>=3
   std::cerr << "alpha = " << libbase::matrix<real>(alpha) << std::endl;
#endif
   }

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::work_beta_and_results(const dev_array1r_t& eof_prior,
      dev_array2r_t& ptable, dev_array1r_t& sof_post, dev_array1r_t& eof_post)
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "Gamma Kernel: (1x" << 2 * xmax + 1 << ") x " << q << std::endl;
      std::cerr << "Beta Kernel: " << 2 * xmax + 1 << " x " << q << std::endl;
      if (computer.flags.norm)
         std::cerr << "Norm. Kernel: " << 1 << " x " << 2 * xmax + 1 << std::endl;
      std::cerr << "Message APP Kernel: " << N << " x " << q << std::endl;
      std::cerr << "State APP Kernel: " << 1 << " x " << 2 * xmax + 1 << std::endl;
      first_time = false;
      }
   // Beta + Results computation:
   for (int i = N; i >= 0; i--)
      {
      // local storage, index value where we need gamma values
      if (!computer.flags.globalstore && i < N)
         {
         // pre-compute local gamma values, if necessary
         if (!computer.flags.lazy)
            {
            // global pre-computation of gamma values
            // block index is for (i,x) where:
            //   i is not used: x-grid size = 1
            //   x in [-xmax, xmax]: y-grid size = 2*xmax+1
            // thread index is for d in [0, q-1]: block size = q
            dim3 gridSize(1, 2*xmax+1);
            fba2_gamma_kernel<receiver_t, sig, real, real2> <<<gridSize,q>>>(dev_object, dev_r, dev_app, i);
            }
         // reset local cache, if necessary
         else
            {
            gamma.local.fill(real(0));
            cached.local.fill(false);
            }
         }
      // block index is for x2 in [-xmax, xmax]: grid size = 2*xmax+1
      // thread index is for d in [0, q-1]: block size = q
      // shared memory: array of q 'real's
      fba2_beta_kernel<receiver_t, sig, real, real2> <<<2*xmax+1,q,q*sizeof(real)>>>(dev_object, eof_prior, i);
      // normalize if requested
      if (computer.flags.norm)
         {
         // NOTE: this has to be done in one block, as we need to sync after
         //       determining the scale to use
         // block index is not used: grid size = 1
         // thread index is for x2 in [-xmax, xmax]: block size = 2*xmax+1
         fba2_normalize_beta_kernel <receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, i);
         }
      // compute partial result
      if (i < N)
         {
         // compute APPs of message
         // block index is not used: grid size = 1
         // thread index is for d in [0, q-1]: block size = q
         fba2_message_app_kernel<receiver_t, sig, real, real2> <<<1,q>>>(dev_object, ptable, i);
         }
      }
   // compute APPs of sof/eof state values
   // block index is not used: grid size = 1
   // thread index is for x in [-xmax, xmax]: block size = 2*xmax+1
   fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, sof_post, 0);
   fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, eof_post, N);

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

template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::init(int N, int n, int q, int I, int xmax,
      int dxmax, double th_inner, double th_outer, bool norm, bool batch,
      bool lazy, bool globalstore)
   {
   // if any parameters that effect memory have changed, release memory
   if (initialised && (N != computer.N || n != computer.n || q != computer.q
         || I != computer.I || xmax != computer.xmax || dxmax != computer.dxmax
         || lazy != computer.flags.lazy || globalstore
         != computer.flags.globalstore))
      {
      free();
      }
   // code parameters
   assert(N > 0);
   assert(n > 0);
   computer.N = N;
   computer.n = n;
   assert(q > 1);
   computer.q = q;
   // decoder parameters
   assert(I >= 0);
   assert(xmax >= 0);
   assert(dxmax >= 0);
   computer.I = I;
   computer.xmax = xmax;
   computer.dxmax = dxmax;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   computer.th_inner = th_inner;
   computer.th_outer = th_outer;
   // decoding mode parameters
   computer.flags.norm = norm;
   computer.flags.batch = batch;
   computer.flags.lazy = lazy;
   computer.flags.globalstore = globalstore;
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
 *       xmax and padded to a total length of tau + 2*xmax, where tau is the
 *       length of the transmitted frame. This avoids special handling for such
 *       vectors.
 *
 * \note Offset is the same as for stream_modulator.
 */
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::decode(libcomm::instrumented& collector,
      const array1s_t& r, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vr_t& ptable,
      array1r_t& sof_post, array1r_t& eof_post, const int offset)
   {
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << computer.N << std::endl;
   std::cerr << "n = " << computer.n << std::endl;
   std::cerr << "q = " << computer.q << std::endl;
   std::cerr << "I = " << computer.I << std::endl;
   std::cerr << "xmax = " << computer.xmax << std::endl;
   std::cerr << "dxmax = " << computer.dxmax << std::endl;
   std::cerr << "th_inner = " << computer.th_inner << std::endl;
   std::cerr << "th_outer = " << computer.th_outer << std::endl;
   std::cerr << "norm = " << computer.flags.norm << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   // Initialise memory on device if necessary
   if (!initialised)
      allocate();
   // Validate sizes and offset
   const int tau = computer.N * computer.n;
   assertalways(offset == computer.xmax);
   assertalways(r.size() == tau + 2 * computer.xmax);
   assertalways(sof_prior.size() == 2 * computer.xmax + 1);
   assertalways(eof_prior.size() == 2 * computer.xmax + 1);

   // Setup device
   gputimer ts("t_setup");
   // copy input data to device, allocating space as needed
   dev_r = r;
   copy_table(app, dev_app);
   dev_sof_table = array1r_t(sof_prior);
   dev_eof_table = array1r_t(eof_prior);
   // allocate space on device for result
   dev_ptable.init(computer.N, computer.q);
   // create a copy of the device object (to pass to kernels)
   dev_object = computer;
   collector.add_timer(ts);
#if DEBUG>=3
   // show input data, as on device
   std::cerr << "r = " << array1s_t(dev_r) << std::endl;
   std::cerr << "app = " << array2r_t(dev_app) << std::endl;
   std::cerr << "sof_prior = " << array1r_t(dev_sof_table) << std::endl;
   std::cerr << "eof_prior = " << array1r_t(dev_eof_table) << std::endl;
#endif

   // Gamma
   if (!computer.flags.lazy && computer.flags.globalstore)
      {
      // compute immediately for global pre-compute mode
      gputimer tg("t_gamma");
      work_gamma(dev_r, dev_app);
      collector.add_timer(tg);
      }
   else
      {
      // keep a copy of received vector and a-priori statistics
      // (we need them later when computing gamma lazily or locally)
      computer.r = dev_r;
      computer.app = dev_app;
      // re-create a copy of the device object (to pass to kernels)
      dev_object = computer;
      // reset cache values if necessary
      if (computer.flags.lazy)
         reset_cache();
      }
   // Alpha + Beta + Results
   if (computer.flags.globalstore)
      {
      // Alpha + Beta
      gputimer tab("t_alpha+beta");
      work_alpha_and_beta(dev_sof_table, dev_eof_table);
      collector.add_timer(tab);
      // Compute results
      gputimer tr("t_results");
      work_results(dev_ptable, dev_sof_table, dev_eof_table);
      collector.add_timer(tr);
      }
   else
      {
      // Alpha
      gputimer ta("t_alpha");
      work_alpha(dev_sof_table);
      collector.add_timer(ta);
      // Beta
      gputimer tbr("t_beta+results");
      work_beta_and_results(dev_eof_table, dev_ptable, dev_sof_table, dev_eof_table);
      collector.add_timer(tbr);
      }
   // Results transfer
   gputimer tc("t_transfer");
   copy_table(dev_ptable, ptable);
   sof_post = array1r_t(dev_sof_table);
   eof_post = array1r_t(dev_eof_table);
   collector.add_timer(tc);

   // add values for limits that depend on channel conditions
   collector.add_timer(computer.I, "c_I");
   collector.add_timer(computer.xmax, "c_xmax");
   collector.add_timer(computer.dxmax, "c_dxmax");
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
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::get_drift_pdf(array1r_t& pdf, const int i) const
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "State APP Kernel: " << 1 << " blocks x " << 2 * xmax + 1
            << " threads" << std::endl;
      first_time = false;
      }
   // Drift PDF computation:
   assert(i >= 0 && i <= N);
   // block index is not used: grid size = 1
   // thread index is for x in [-xmax, xmax]: block size = 2*xmax+1
   fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, dev_sof_table, i);
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
template <class receiver_t, class sig, class real, class real2>
void fba2<receiver_t, sig, real, real2>::get_drift_pdf(array1vr_t& pdftable) const
   {
   assert( initialised);
   // Shorthand
   const int N = computer.N;
   const int q = computer.q;
   const int xmax = computer.xmax;
   // inform user what the kernel sizes are
   static bool first_time = true;
   if (first_time)
      {
      std::cerr << "State APP Kernel (x" << N + 1 << "): " << 1 << " blocks x "
            << 2 * xmax + 1 << " threads" << std::endl;
      first_time = false;
      }
   // Drift PDF computation:
   // allocate space for results
   pdftable.init(N + 1);
   // consider each time index in the order given
   for (int i = 0; i <= N; i++)
      {
      // block index is not used: grid size = 1
      // thread index is for x in [-xmax, xmax]: block size = 2*xmax+1
      fba2_state_app_kernel<receiver_t, sig, real, real2> <<<1,2*xmax+1>>>(dev_object, dev_sof_table, i);
      // copy result from temporary space
      pdftable(i) = array1r_t(dev_sof_table);
      }
   }

} // end namespace

// Explicit Realizations

#include "modem/tvb-receiver-cuda.h"
#include "gf.h"

namespace cuda {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)
#define REAL2_TYPE_SEQ \
   (float)(double)

// *** Instantiations for tvb: gf types only ***

#define INSTANTIATE_TVB(r, args) \
      template class fba2<tvb_receiver<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)> ; \
      template class value<fba2<tvb_receiver<BOOST_PP_SEQ_ENUM(args)> , \
         BOOST_PP_SEQ_ENUM(args)>::metric_computer> ; \

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_TVB, (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ)(REAL2_TYPE_SEQ))

} // end namespace
