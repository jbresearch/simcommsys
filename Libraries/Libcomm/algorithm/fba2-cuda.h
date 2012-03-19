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

template <class receiver_t, class real, class sig, bool norm>
class fba2 {
public:
   /*! \name Type definitions */
   // Device-based types
   typedef cuda::vector<sig> dev_array1s_t;
   typedef cuda::vector<real> dev_array1r_t;
   typedef cuda::matrix<real> dev_array2r_t;
   typedef cuda::vector_reference<sig> dev_array1s_ref_t;
   typedef cuda::vector_reference<real> dev_array1r_ref_t;
   typedef cuda::matrix_reference<real> dev_array2r_ref_t;
   // Host-based types
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   // @}
private:
   /*! \name User-defined parameters */
   int N; //!< The transmitted block size in symbols
   int n; //!< The number of bits encoding each q-ary symbol
   int q; //!< The number of symbols in the q-ary alphabet
   int I; //!< The maximum number of insertions considered before every transmission
   int xmax; //!< The maximum allowed overall drift is \f$ \pm x_{max} \f$
   int dxmax; //!< The maximum allowed drift within a q-ary symbol is \f$ \pm \delta_{max} \f$
   real th_inner; //!< Threshold factor for inner cycle
   real th_outer; //!< Threshold factor for outer cycle
   // @}
   /*! \name Internally-used objects */
   int dmin; //!< Offset for deltax index in gamma matrix
   int dmax; //!< Maximum value for deltax index in gamma matrix
   dev_array2r_ref_t alpha; //!< Forward recursion metric
   dev_array2r_ref_t beta; //!< Backward recursion metric
   dev_array1r_ref_t gamma; //!< Receiver metric
   mutable receiver_t receiver; //!< Inner code receiver metric computation
   // @}
private:
   /*! \name Internal functions */
#ifdef __CUDACC__
   __device__ __host__
#endif
   int get_gamma_index(int d, int i, int x, int deltax) const
      {
      // gamma has indices (d,i,x,deltax) where:
      //    d in [0, q-1], i in [0, N-1], x in [-xmax, xmax], and
      //    deltax in [dmin, dmax] = [max(-n,-xmax), min(nI,xmax)]
      const int pitch3 = (dmax - dmin + 1);
      const int pitch2 = pitch3 * (2 * xmax + 1);
      const int pitch1 = pitch2 * N;
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
   __device__
   real get_gamma(int d, int i, int x, int deltax) const
      {
      return gamma(get_gamma_index(d, i, x, deltax));
      }
   __device__
   real& get_gamma(int d, int i, int x, int deltax)
      {
      return gamma(get_gamma_index(d, i, x, deltax));
      }
#endif
   // memory allocation
   void allocate(dev_array2r_t& alpha, dev_array2r_t& beta,
         dev_array1r_t& gamma);
   // helper methods
   void print_gamma(std::ostream& sout) const;
   // de-reference kernel calls
#ifdef __CUDACC__
   void do_work_gamma(const dev_array1s_t& r, const dev_array2r_t& app);
   void do_work_alpha(const dev_array1r_t& sof_prior, stream& sid);
   void do_work_beta(const dev_array1r_t& eof_prior, stream& sid);
   void do_work_results(dev_array2r_t& ptable, dev_array1r_t& sof_post,
         dev_array1r_t& eof_post) const;
#endif
   void copy_results(const dev_array2r_t& dev_ptable, array1vr_t& ptable);
   // @}
public:
   /*! \name Internal functions */
   // Device-only methods
#ifdef __CUDACC__
   __device__
   void work_gamma(const dev_array1s_ref_t& r, const dev_array2r_ref_t& app);
   __device__
   static real get_threshold(const dev_array2r_ref_t& metric, int row, int cols, real factor);
   __device__
   static real parallel_sum(real array[]);
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
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2()
      {
      }
   // @}

   // main initialization routine
   void init(int N, int n, int q, int I, int xmax, int dxmax, double th_inner,
         double th_outer);

   /*! \name Parameter getters */
   //! Access metric computation
   receiver_t& get_receiver() const
      {
      return receiver;
      }
   int get_N() const
      {
      return N;
      }
   int get_n() const
      {
      return n;
      }
   int get_q() const
      {
      return q;
      }
   int get_I() const
      {
      return I;
      }
   int get_xmax() const
      {
      return xmax;
      }
   int get_dxmax() const
      {
      return dxmax;
      }
   double get_th_inner() const
      {
      return th_inner;
      }
   double get_th_outer() const
      {
      return th_outer;
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
