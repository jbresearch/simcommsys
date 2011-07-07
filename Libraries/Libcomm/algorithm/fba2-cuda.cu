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

// Memory allocation

/*! \brief Memory allocator for working matrices
 */
template <class real, class sig, bool norm>
void fba2<real, sig, norm>::allocate(dev_array2r_t& alpha, dev_array2r_t& beta,
      dev_array1r_t& gamma)
   {
   // determine limits
   dmin = std::max(-n, -dxmax);
   dmax = std::min(n * I, dxmax);
   // determine required space for inner metric caching
#ifdef DEBUG
   size_t entries = 0;
   for (int delta = dmin; delta <= dmax; delta++)
      entries += (1 << (delta + n));
   std::cerr << "Inner Metric: " << q * entries * sizeof(float) / double(1
         << 20) << "MiB" << std::endl;
#endif
   // alpha needs indices (i,x) where i in [0, N-1] and x in [-xmax, xmax]
   // beta needs indices (i,x) where i in [1, N] and x in [-xmax, xmax]
   alpha.init(N, 2 * xmax + 1); // offsets: 0, xmax
   beta.init(N, 2 * xmax + 1); // offsets: -1, xmax
   // gamma needs indices (d,i,x,deltax) where d in [0, q-1], i in [0, N-1]
   // x in [-xmax, xmax], and deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
   gamma.init(q * N * (2 * xmax + 1) * (dmax - dmin + 1));
   // copy over to references
   this->alpha = alpha;
   this->beta = beta;
   this->gamma = gamma;

   // if this is not the first time, skip the rest
   static bool first_time = true;
   if (!first_time)
      return;
   first_time = false;

   // set required format, storing previous settings
   const std::ios::fmtflags flags = std::cerr.flags();
   std::cerr.setf(std::ios::fixed, std::ios::floatfield);
   const int prec = std::cerr.precision(1);
   // determine memory occupied and tell user
   const size_t bytes_used = sizeof(real) * (alpha.size() + beta.size()
         + gamma.size());
   std::cerr << "FBA Memory Usage: " << bytes_used / double(1 << 20) << "MiB"
         << std::endl;
   // revert cerr to original format
   std::cerr.precision(prec);
   std::cerr.setf(flags);

#if DEBUG>=2
   std::cerr << "Allocated FBA memory..." << std::endl;
   std::cerr << "dmax = " << dmax << std::endl;
   std::cerr << "dmin = " << dmin << std::endl;
   std::cerr << "alpha = " << N << "x" << 2 * xmax + 1 << " = " << alpha.size()
         << std::endl;
   std::cerr << "beta = " << N << "x" << 2 * xmax + 1 << " = " << beta.size()
         << std::endl;
   std::cerr << "gamma = " << q << "x" << N << "x" << 2 * xmax + 1 << "x"
         << dmax - dmin + 1 << " = " << gamma.size() << std::endl;
#endif
   }

// Initialization

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::init(int N, int n, int q, int I, int xmax,
      int dxmax, double th_inner, double th_outer)
   {
   // code parameters
   assert(N > 0);
   assert(n > 0);
   this->N = N;
   this->n = n;
   assert(q > 1);
   this->q = q;
   // decoder parameters
   assert(I > 0);
   assert(xmax > 0);
   assert(dxmax > 0);
   this->I = I;
   this->xmax = xmax;
   this->dxmax = dxmax;
   // path truncation parameters
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   this->th_inner = th_inner;
   this->th_outer = th_outer;
   }

// Internal procedures

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::work_gamma(const dev_array1s_ref_t& r,
      const dev_array2r_ref_t& app)
   {
   using cuda::min;
   using cuda::max;

   if (app.size() == 0)
      {
      work_gamma(r);
      return;
      }
   // length of received sequence
   const int rho = r.size();
   // set up block & thread indexes
   const int i = blockIdx.x;
   const int d = threadIdx.x;
   // compute all matrix values
   // - all threads are independent and indexes guaranteed in range

   // event must fit the received sequence:
   // (this is limited to start and end conditions)
   // 1. n*i+x1 >= 0
   // 2. n*(i+1)-1+x2 <= rho-1
   // limits on insertions and deletions must be respected:
   // 3. x2-x1 <= n*I
   // 4. x2-x1 >= -n
   // limits on introduced drift in this section:
   // (necessary for forward recursion on extracted segment)
   // 5. x2-x1 <= dxmax
   // 6. x2-x1 >= -dxmax
   const int x1min = max(-xmax, -n * i);
   const int x1max = xmax;
   for (int x1 = x1min; x1 <= x1max; x1++)
      {
      const int deltaxmin = max(-xmax - x1, dmin);
      const int deltaxmax = min(min(xmax - x1, dmax), rho - n * (i + 1) - x1);
      for (int deltax = deltaxmin; deltax <= deltaxmax; deltax++)
         {
         real R = receiver.R(d, i, r.extract(n * i + x1, n + deltax));
         R *= app(i, d);
         get_gamma(d, i, x1, deltax) = R;
         }
      }
   }

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::work_gamma(const dev_array1s_ref_t& r)
   {
   using cuda::min;
   using cuda::max;

   // length of received sequence
   const int rho = r.size();
   // set up block & thread indexes
   const int i = blockIdx.x;
   const int d = threadIdx.x;
   // - all threads are independent and indexes guaranteed in range
   // compute all matrix values

   // event must fit the received sequence:
   // (this is limited to start and end conditions)
   // 1. n*i+x1 >= 0
   // 2. n*(i+1)-1+x2 <= rho-1
   // limits on insertions and deletions must be respected:
   // 3. x2-x1 <= n*I
   // 4. x2-x1 >= -n
   // limits on introduced drift in this section:
   // (necessary for forward recursion on extracted segment)
   // 5. x2-x1 <= dxmax
   // 6. x2-x1 >= -dxmax
   const int x1min = max(-xmax, -n * i);
   const int x1max = xmax;
   for (int x1 = x1min; x1 <= x1max; x1++)
      {
      const int x2min = max(-xmax, dmin + x1);
      const int x2max = min(min(xmax, dmax + x1), rho - n * (i + 1));
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         real R = receiver.R(d, i, r.extract(n * i + x1, n + x2 - x1));
         get_gamma(d, i, x1, x2 - x1) = R;
         }
      }
   }

template <class real, class sig, bool norm>
__device__
real fba2<real, sig, norm>::get_threshold(const dev_array2r_ref_t& metric, int row, int cols, real factor)
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

template <class real, class sig, bool norm>
__device__
real fba2<real, sig, norm>::parallel_sum(real array[])
   {
   const int N = blockDim.x; // Total number of active threads
   const int i = threadIdx.x;
   cuda_assert(N % 2 == 0);
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

template <class real, class sig, bool norm>
__device__
real fba2<real, sig, norm>::get_scale(const dev_array2r_ref_t& metric, int row, int cols)
   {
   real scale = 0;
   for (int col = 0; col < cols; col++)
      {
      scale += metric(row, col);
      }
   cuda_assertalways(scale > real(0));
   scale = real(1) / scale;
   // wait until all threads have completed their part
   __syncthreads();
   return scale;
   }

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::normalize(dev_array2r_ref_t& metric, int row, int cols)
   {

   // set up thread index
   const int col = threadIdx.x;
   // determine the scale factor to use (each block has to do this)
   const real scale = get_scale(metric, row, cols);
   // scale all results
   metric(row, col) *= scale;
   }

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::work_alpha(int rho, int i)
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_inner > 0);
   // set up block & thread indexes
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
         // we know x[0] = 0; ie. drift before transmitting bit t0 is zero.
         alpha(0, x2 + xmax) = real(x2 == 0);
         }
      }
   else
      {
      // compute remaining matrix values:
      // determine the strongest path at this point
      const real threshold = get_threshold(alpha, i - 1, 2 * xmax + 1, th_inner);
      // initialize result holder
      this_alpha[d] = 0;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*(i-1)+x1 >= 0
      // 2. n*i-1+x2 <= rho-1
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= dxmax
      // 6. x2-x1 >= -dxmax
      const int x1min = max(-xmax, -n * (i - 1));
      const int x1max = xmax;
      for (int x1 = x1min; x1 <= x1max; x1++)
         {
         // cache previous alpha value in a register
         real prev_alpha = alpha(i - 1, x1 + xmax);
         // ignore paths below a certain threshold
         if (!thresholding || prev_alpha >= threshold)
            {
            // each block computes for a different end-state (x2)
            const int x2min = max(-xmax, dmin + x1);
            const int x2max = min(min(xmax, dmax + x1), rho - n * i);
            if (x2 >= x2min && x2 <= x2max)
               {
               // each thread in a block is computing for a different 'd'
               real temp = prev_alpha;
               temp *= get_gamma(d, i - 1, x1, x2 - x1);
               this_alpha[d] += temp;
               }
            }
         }
      // make sure all threads in block have finished updating this_alpha
      __syncthreads();
      // compute sum of shared array
      const real temp = parallel_sum(this_alpha);
      // store result (first thread in block)
      if (d == 0)
         {
         alpha(i, x2 + xmax) = temp;
         }
      }
   }

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::work_beta(int rho, int i)
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_inner > 0);
   // set up block & thread indexes
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
         // we know x[tau] = rho-tau; ie. drift before transmitting bit t[tau] is
         // the discrepancy in the received vector size from tau
         const int tau = N * n;
         cuda_assertalways(abs(rho - tau) <= xmax);
         beta(N - 1, x1 + xmax) = real(x1 == rho - tau);
         }
      }
   else
      {
      // compute remaining matrix values:
      // determine the strongest path at this point
      const real threshold = get_threshold(beta, i + 1 - 1, 2 * xmax + 1, th_inner);
      // initialize result holder
      this_beta[d] = 0;
      // event must fit the received sequence:
      // (this is limited to start and end conditions)
      // 1. n*i+x1 >= 0
      // 2. n*(i+1)-1+x2 <= rho-1
      // limits on insertions and deletions must be respected:
      // 3. x2-x1 <= n*I
      // 4. x2-x1 >= -n
      // limits on introduced drift in this section:
      // (necessary for forward recursion on extracted segment)
      // 5. x2-x1 <= dxmax
      // 6. x2-x1 >= -dxmax
      const int x2min = -xmax;
      const int x2max = min(xmax, rho - n * (i + 1));
      for (int x2 = x2min; x2 <= x2max; x2++)
         {
         // cache next beta value in a register
         real next_beta = beta(i + 1 - 1, x2 + xmax);
         // ignore paths below a certain threshold
         if (!thresholding || next_beta >= threshold)
            {
            // each block computes for a different start-state (x1)
            const int x1min = max(max(-xmax, x2 - dmax), -n * i);
            const int x1max = min(xmax, x2 - dmin);
            if (x1 >= x1min && x1 <= x1max)
               {
               // each thread in a block is computing for a different 'd'
               real temp = next_beta;
               temp *= get_gamma(d, i, x1, x2 - x1);
               this_beta[d] += temp;
               }
            }
         }
      // make sure all threads in block have finished updating this_beta
      __syncthreads();
      // compute sum of shared array
      const real temp = parallel_sum(this_beta);
      // store result
      if (d == 0)
         {
         beta(i - 1, x1 + xmax) = temp;
         }
      }
   }

template <class real, class sig, bool norm>
__device__
void fba2<real, sig, norm>::work_results(int rho, dev_array2r_ref_t& ptable) const
   {
   using cuda::min;
   using cuda::max;

   // local flag for path thresholding
   const bool thresholding = (th_outer > 0);
   // Check result vector (one sparse symbol per timestep)
   cuda_assertalways(ptable.get_rows()==N && ptable.get_cols()==q);
   // set up block & thread indexes
   const int i = blockIdx.x;
   const int d = threadIdx.x;
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   // - all threads are independent and indexes guaranteed in range
   // determine the strongest path at this point
   const real threshold = get_threshold(alpha, i, 2 * xmax + 1, th_outer);
   // initialize result holder
   real p = 0;
   // event must fit the received sequence:
   // (this is limited to start and end conditions)
   // 1. n*i+x1 >= 0
   // 2. n*(i+1)-1+x2 <= rho-1
   // limits on insertions and deletions must be respected:
   // 3. x2-x1 <= n*I
   // 4. x2-x1 >= -n
   // limits on introduced drift in this section:
   // (necessary for forward recursion on extracted segment)
   // 5. x2-x1 <= dxmax
   // 6. x2-x1 >= -dxmax
   const int x1min = max(-xmax, -n * i);
   const int x1max = xmax;
   for (int x1 = x1min; x1 <= x1max; x1++)
      {
      // cache this alpha value in a register
      real this_alpha = alpha(i, x1 + xmax);
      // ignore paths below a certain threshold
      if (!thresholding || this_alpha >= threshold)
         {
         const int x2min = max(-xmax, dmin + x1);
         const int x2max = min(min(xmax, dmax + x1), rho - n * (i + 1));
         for (int x2 = x2min; x2 <= x2max; x2++)
            {
            real temp = this_alpha;
            temp *= beta(i + 1 - 1, x2 + xmax);
            temp *= get_gamma(d, i, x1, x2 - x1);
            p += temp;
            }
         }
      }
   // store result
   ptable(i,d) = p;
   }

// Kernels

template <class real, class sig, bool norm>
__global__
void fba2_gamma_kernel(fba2<real,sig,norm> object, const vector_reference<sig> r,
      const matrix_reference<real> app)
   {
   object.work_gamma(r, app);
   }

template <class real, class sig, bool norm>
__global__
void fba2_gamma_kernel(fba2<real,sig,norm> object, const vector_reference<sig> r)
   {
   object.work_gamma(r);
   }

template <class real, class sig, bool norm>
__global__
void fba2_alpha_kernel(fba2<real,sig,norm> object, const int rho, const int i)
   {
   object.work_alpha(rho, i);
   }

template <class real, class sig, bool norm>
__global__
void fba2_normalize_alpha_kernel(fba2<real,sig,norm> object, const int i)
   {
   object.normalize_alpha(i);
   }

template <class real, class sig, bool norm>
__global__
void fba2_beta_kernel(fba2<real,sig,norm> object, const int rho, const int i)
   {
   object.work_beta(rho, i);
   }

template <class real, class sig, bool norm>
__global__
void fba2_normalize_beta_kernel(fba2<real,sig,norm> object, const int i)
   {
   object.normalize_beta(i);
   }

template <class real, class sig, bool norm>
__global__
void fba2_results_kernel(fba2<real,sig,norm> object, const int rho,
      matrix_reference<real> ptable)
   {
   object.work_results(rho, ptable);
   }

// helper methods

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::print_gamma(std::ostream& sout) const
   {
   // copy the data set from the device
   libbase::vector<real> host_gamma = libbase::vector<real>(gamma);
   // gamma has indices (d,i,x,deltax) where:
   //    d in [0, q-1], i in [0, N-1], x in [-xmax, xmax], and
   //    deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
   for (int i = 0; i < N; i++)
      {
      sout << "i = " << i << ":" << std::endl;
      for (int d = 0; d < q; d++)
         {
         sout << "d = " << d << ":" << std::endl;
         for (int x = -xmax; x <= xmax; x++)
            {
            for (int deltax = dmin; deltax <= dmax; deltax++)
               {
               const int ndx = get_gamma_index(d, i, x, deltax);
               sout << '\t' << host_gamma(ndx);
               }
            sout << std::endl;
            }
         }
      }
   }

// de-reference kernel calls

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::do_work_gamma(const dev_array1s_t& dev_r,
      const dev_array2r_t& dev_app)
   {
   static bool first_time = true;
   // Gamma computation:
   if (first_time)
      std::cerr << "Gamma Kernel: " << N << " blocks x " << q << " threads"
            << std::endl;
   // reset gamma values
   gamma.fill(0);
   // block index is for i in [0, N-1]: grid size = N
   // thread index is for d in [0, q-1]: block size = q
   fba2_gamma_kernel<real,sig,norm> <<<N,q>>>(*this, dev_r, dev_app);
   cudaSafeThreadSynchronize();
   // if debug mode is high enough, print the results obtained
#if DEBUG>=3
   std::cerr << "gamma = " << std::endl;
   print_gamma(std::cerr);
#endif
   // reset pacifier monitor
   first_time = false;
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::do_work_gamma(const dev_array1s_t& dev_r)
   {
   static bool first_time = true;
   // Gamma computation:
   if (first_time)
      std::cerr << "Gamma Kernel: " << N << " blocks x " << q << " threads"
            << std::endl;
   // reset gamma values
   gamma.fill(0);
   // block index is for i in [0, N-1]: grid size = N
   // thread index is for d in [0, q-1]: block size = q
   fba2_gamma_kernel<real,sig,norm> <<<N,q>>>(*this, dev_r);
   cudaSafeThreadSynchronize();
   // if debug mode is high enough, print the results obtained
#if DEBUG>=3
   std::cerr << "gamma = " << std::endl;
   print_gamma(std::cerr);
#endif
   // reset pacifier monitor
   first_time = false;
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::do_work_alpha(int rho, stream& sid)
   {
   static bool first_time = true;
   // Alpha computation:
   if (first_time)
      {
      std::cerr << "Alpha Kernel: " << 2 * xmax + 1 << " blocks x " << q
            << " threads" << std::endl;
      if (norm)
         std::cerr << "Normalization Kernel: " << 1 << " blocks x " << 2 * xmax
               + 1 << " threads" << std::endl;
      }
   for (int i = 0; i < N; i++)
      {
      // block index is for x2 in [-xmax, xmax]: grid size = 2*xmax+1
      // thread index is for d in [0, q-1]: block size = q
      // shared memory: array of q 'real's
      fba2_alpha_kernel<real,sig,norm> <<<2*xmax+1,q,q*sizeof(real),sid.get_id()>>>(*this, rho, i);
      //cudaSafeThreadSynchronize();
      // normalize if requested
      if (norm)
         {
         // block index is not used: grid size = 1
         // thread index is for x2 in [-xmax, xmax]: block size = 2*xmax+1
fba2_normalize_alpha_kernel         <real,sig,norm> <<<1,2*xmax+1,0,sid.get_id()>>>(*this, i);
         //cudaSafeThreadSynchronize();
         }
      }
   // if debug mode is high enough, print the results obtained
#if DEBUG>=3
   std::cerr << "alpha = " << libbase::matrix<real>(alpha) << std::endl;
#endif
   // reset pacifier monitor
   first_time = false;
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::do_work_beta(int rho, stream& sid)
   {
   static bool first_time = true;
   // Beta computation:
   if (first_time)
      {
      std::cerr << "Beta Kernel: " << 2 * xmax + 1 << " blocks x " << q
      << " threads" << std::endl;
      if (norm)
      std::cerr << "Normalization Kernel: " << 1 << " blocks x " << 2 * xmax
      + 1 << " threads" << std::endl;
      }
   for (int i = N; i > 0; i--)
      {
      // block index is for x2 in [-xmax, xmax]: grid size = 2*xmax+1
      // thread index is for d in [0, q-1]: block size = q
      // shared memory: array of q 'real's
      fba2_beta_kernel<real,sig,norm> <<<2*xmax+1,q,q*sizeof(real),sid.get_id()>>>(*this, rho, i);
      //cudaSafeThreadSynchronize();
      // normalize if requested
      if (norm)
         {
         // block index is not used: grid size = 1
         // thread index is for x2 in [-xmax, xmax]: block size = 2*xmax+1
         fba2_normalize_beta_kernel<real,sig,norm> <<<1,2*xmax+1,0,sid.get_id()>>>(*this, i);
         //cudaSafeThreadSynchronize();
         }
      }
   // if debug mode is high enough, print the results obtained
#if DEBUG>=3
   std::cerr << "beta = " << libbase::matrix<real>(beta) << std::endl;
#endif
   // reset pacifier monitor
   first_time = false;
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::do_work_results(int rho, dev_array2r_t& dev_ptable) const
   {
   static bool first_time = true;
   // Results computation:
   // block index is for i in [0, N-1]: grid size = N
   // thread index is for d in [0, q-1]: block size = q
   if (first_time)
   std::cerr << "Results Kernel: " << N << " blocks x " << q << " threads"
   << std::endl;
   fba2_results_kernel<real,sig,norm> <<<N,q>>>(*this, rho, dev_ptable);
   cudaSafeThreadSynchronize();
   // reset pacifier monitor
   first_time = false;
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::copy_results(const dev_array2r_t& dev_ptable,
      array1vr_t& ptable)
   {
   // initialise result vector (one sparse symbol per timestep) and copy back
   libbase::allocate(ptable, N, q);
   for (int i = 0; i < N; i++)
   ptable(i) = array1r_t(dev_ptable.extract_row(i));
   // if debug mode is high enough, print the results obtained
#if DEBUG>=3
   std::cerr << "ptable = " << ptable << std::endl;
#endif
   }

// User procedures

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::decode(libcomm::instrumented& collector,
      const array1s_t& r, const array1vd_t& app, array1vr_t& ptable)
   {
   // allocate memory on device
   dev_array2r_t alpha;
   dev_array2r_t beta;
   dev_array1r_t gamma;
   allocate(alpha, beta, gamma);
   // allocate space on device for rx vector, and copy over
   dev_array1s_t dev_r;
   dev_r = r;
   // allocate space on device for app table, and copy over if necessary
   dev_array2r_t dev_app;
   if (app.size() > 0)
      {
      dev_app.init(N, q);
      assert(app.size() == N);
      for (int i = 0; i < N; i++)
         {
         assert(app(i).size() == q);
         dev_app.extract_row(i) = array1r_t(app(i));
         }
      }
   // allocate space on device for result, and initialize
   dev_array2r_t dev_ptable;
   dev_ptable.init(N, q);
   // call the kernels
   gputimer tg("t_gamma_app");
   do_work_gamma(dev_r, dev_app);
   collector.add_timer(tg);

   gputimer tab("t_alpha+beta");
   stream sa, sb;
   gputimer ta("t_alpha", sa.get_id());
   do_work_alpha(r.size(), sa);
   ta.stop();
   gputimer tb("t_beta", sb.get_id());
   do_work_beta(r.size(), sb);
   tb.stop();
   cudaSafeThreadSynchronize();
   collector.add_timer(ta);
   collector.add_timer(tb);
   collector.add_timer(tab);

   gputimer tr("t_results");
   do_work_results(r.size(), dev_ptable);
   collector.add_timer(tr);

   gputimer tc("t_transfer");
   copy_results(dev_ptable, ptable);
   collector.add_timer(tc);
   // add values for limits that depend on channel conditions
   collector.add_timer(I, "c_I");
   collector.add_timer(xmax, "c_xmax");
   collector.add_timer(dxmax, "c_dxmax");
   // add memory usage
   collector.add_timer(sizeof(real) * alpha.size(), "m_alpha"); 
   collector.add_timer(sizeof(real) * beta.size(), "m_beta"); 
   collector.add_timer(sizeof(real) * gamma.size(), "m_gamma"); 
   }

template <class real, class sig, bool norm>
void fba2<real, sig, norm>::decode(libcomm::instrumented& collector,
      const array1s_t& r, array1vr_t& ptable)
   {
   // allocate memory on device
   dev_array2r_t alpha;
   dev_array2r_t beta;
   dev_array1r_t gamma;
   allocate(alpha, beta, gamma);
#if DEBUG>=3
   std::cerr << "Starting decode..." << std::endl;
   std::cerr << "N = " << N << std::endl;
   std::cerr << "n = " << n << std::endl;
   std::cerr << "q = " << q << std::endl;
   std::cerr << "I = " << I << std::endl;
   std::cerr << "xmax = " << xmax << std::endl;
   std::cerr << "dxmax = " << dxmax << std::endl;
   std::cerr << "th_inner = " << th_inner << std::endl;
   std::cerr << "th_outer = " << th_outer << std::endl;
   std::cerr << "norm = " << norm << std::endl;
   std::cerr << "real = " << typeid(real).name() << std::endl;
#endif
   // allocate space on device for rx vector, and copy over
   dev_array1s_t dev_r;
   dev_r = r;
#if DEBUG>=3
   std::cerr << "r = " << array1s_t(dev_r) << std::endl;
#endif
   // allocate space on device for result, and initialize
   dev_array2r_t dev_ptable;
   dev_ptable.init(N, q);
   // call the kernels
   gputimer tg("t_gamma");
   do_work_gamma(dev_r);
   collector.add_timer(tg);

   gputimer tab("t_alpha+beta");
   stream sa, sb;
   gputimer ta("t_alpha", sa.get_id());
   do_work_alpha(r.size(), sa);
   ta.stop();
   gputimer tb("t_beta", sb.get_id());
   do_work_beta(r.size(), sb);
   tb.stop();
   cudaSafeThreadSynchronize();
   collector.add_timer(ta);
   collector.add_timer(tb);
   collector.add_timer(tab);

   gputimer tr("t_results");
   do_work_results(r.size(), dev_ptable);
   collector.add_timer(tr);

   gputimer tc("t_transfer");
   copy_results(dev_ptable, ptable);
   collector.add_timer(tc);
   // add values for limits that depend on channel conditions
   collector.add_timer(I, "c_I");
   collector.add_timer(xmax, "c_xmax");
   collector.add_timer(dxmax, "c_dxmax");
   // add memory usage
   collector.add_timer(sizeof(real) * alpha.size(), "m_alpha"); 
   collector.add_timer(sizeof(real) * beta.size(), "m_beta"); 
   collector.add_timer(sizeof(real) * gamma.size(), "m_gamma"); 
   }

// Explicit Realizations

// no-normalization, for debugging
template class fba2<float, bool, false>;
template class fba2<double, bool, false>;
// normalized, for normal use
template class fba2<float, bool, true>;
template class fba2<double, bool, true>;

} // end namespace
