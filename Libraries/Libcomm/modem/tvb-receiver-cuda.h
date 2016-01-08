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

#ifndef __tvb_receiver_cuda_h
#define __tvb_receiver_cuda_h

#include "config.h"
#include "cuda-all.h"
#include "channel/qids.h"

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show settings when initializing the tvb computer
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Time-Varying Block Code support [CUDA].
 * \author  Johann Briffa
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for returning results
 * \tparam real2 Floating-point type for internal computation
 */

template <class sig, class real, class real2>
class tvb_receiver {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<real> array1r_t;
   // @}
private:
   /*! \name User-defined parameters */
   mutable cuda::vector_auto<sig> encoding_table; //!< Local copy of per-frame encoding table, flattened
   mutable cuda::vector_auto<int> cw_start; //!< Start of segment in encoding table for codeword 'i'
   mutable cuda::vector_auto<int> cw_length; //!< Length of codeword 'i'
   typename libcomm::qids<sig,real2>::metric_computer computer; //!< Channel object for computing receiver metric
   // @}
public:
   /*! \name User initialization (can be adapted for needs of user class) */
   /*! \brief Set up channel receiver
    * Only needs to be done before the first frame.
    */
   void init(const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer)
      {
      this->computer = dynamic_cast<const typename libcomm::qids<sig, real2>::metric_computer&> (computer);
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "T = " << computer.T << std::endl;
      std::cerr << "mT_min = " << computer.mT_min << std::endl;
      std::cerr << "mT_max = " << computer.mT_max << std::endl;
      std::cerr << "m1_min = " << computer.m1_min << std::endl;
      std::cerr << "m1_max = " << computer.m1_max << std::endl;
      std::cerr << "Rval = " << computer.Rval << std::endl;
      std::cerr << "Rtable = " << libbase::matrix<real2>(
            computer.Rtable) << std::endl;
#endif
      }
   /*! \brief Set up encoding table
    * Needs to be done before every frame.
    */
   void init(const array2vs_t& encoding_table) const
      {
      // shorthand
      const int N = encoding_table.size().rows();
      const int q = encoding_table.size().cols();
      // initialize arrays with start of segment and length of each codeword
      array1i_t cw_length; //!< Codeword 'i' length
      array1i_t cw_start; //!< Codeword 'i' segment index in flattened table
      cw_length.init(N);
      cw_start.init(N);
      int start = 0;
      for (int i = 0; i < N; i++)
         {
         const int n = encoding_table(i, 0).size();
         cw_start(i) = start;
         cw_length(i) = n;
         start += n * q;
         }
      // transfer to device
      this->cw_start = cw_start;
      this->cw_length = cw_length;
      // size of flattened table
      const int tauq = start;
      // flatten on host first
      static array1s_t temp;
      temp.init(tauq);
      for (int i = 0; i < N; i++)
         for (int d = 0; d < q; d++)
            temp.segment(cw_start(i) + d * cw_length(i), cw_length(i)) =
                  encoding_table(i, d);
      // copy to device
      this->encoding_table = temp;
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "encoding_table = " << array1s_t(this->encoding_table) << std::endl;
      std::cerr << "sizeof(encoding_table) = " << sizeof(this->encoding_table) << std::endl;
#endif
      }
   // @}
   /*! \name Information functions */
   //! Determine the amount of shared memory required per receiver thread
   size_t receiver_sharedmem() const
      {
      return computer.receiver_sharedmem();
      }
   // @}
#ifdef __CUDACC__
   /*! \name Interface with fba2 algorithm (cannot be changed) */
   //! Batch receiver interface
   __device__
   void R(int d, int i, const cuda::vector_reference<sig>& r,
         cuda::vector_reference<real2>& ptable) const
      {
      const cuda::vector<sig>& tx = encoding_table.extract(cw_start(i) + d *
            cw_length(i), cw_length(i));
      // call batch receiver method
      computer.receive(tx, r, ptable);
      }
   // @}
#endif
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
