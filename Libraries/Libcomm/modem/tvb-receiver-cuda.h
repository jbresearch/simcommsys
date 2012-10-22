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
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for returning results
 * \tparam real2 Floating-point type for internal computation
 */

template <class sig, class real, class real2>
class tvb_receiver {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<real> array1r_t;
   // @}
private:
   /*! \name User-defined parameters */
   int n; //!< Number of bits per codeword
   mutable cuda::matrix_auto<cuda::vector_auto<sig> > encoding_table; //!< Local copy of per-frame encoding table
   typename libcomm::qids<sig,real2>::metric_computer computer; //!< Channel object for computing receiver metric
   // @}
public:
   /*! \name User initialization (can be adapted for needs of user class) */
   /*! \brief Set up code size and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(const int n, const libcomm::qids<sig,real2>& chan)
      {
      this->n = n;
      computer = chan.get_computer();
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "n = " << this->n << std::endl;
      std::cerr << "N = " << computer.N << std::endl;
      std::cerr << "I = " << computer.I << std::endl;
      std::cerr << "xmax = " << computer.xmax << std::endl;
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
      // copy contents to device first
      static libbase::matrix<cuda::vector_auto<sig> > temp;
      temp = encoding_table;
      // now make a table of references on device
      this->encoding_table = temp;
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "encoding_table = " << array2vs_t(this->encoding_table) << std::endl;
      std::cerr << "sizeof(encoding_table) = " << sizeof(this->encoding_table) << std::endl;
#endif
      }
   // @}
#ifdef __CUDACC__
   /*! \name Interface with fba2 algorithm (cannot be changed) */
   //! Receiver interface
   __device__
   real R(int d, int i, const cuda::vector_reference<sig>& r) const
      {
      // 'tx' is the vector of transmitted symbols that we're considering
      const cuda::vector<sig>& tx = encoding_table(i, d);
      // compute the conditional probability
      return computer.receive(tx, r);
      }
   //! Batch receiver interface
   __device__
   void R(int d, int i, const cuda::vector_reference<sig>& r,
         cuda::vector_reference<real2>& ptable) const
      {
      const cuda::vector<sig>& tx = encoding_table(i, d);
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
