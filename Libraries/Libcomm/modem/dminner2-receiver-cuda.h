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

#ifndef __dminner2_receiver_cuda_h
#define __dminner2_receiver_cuda_h

#include "config.h"
#include "cuda-all.h"
#include "channel/bsid.h"

namespace cuda {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show settings when initializing the dminner2 computer
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Davey-MacKay Inner Code support [CUDA].
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class real>
class dminner2_receiver {
private:
   int n; //!< Number of bits per codeword
   mutable cuda::vector<int> marker; //!< Device copy of marker sequence
   cuda::matrix_auto<int> codebook; //!< Device copy of codebook
   libcomm::bsid::metric_computer computer; //!< Channel object for computing receiver metric
public:
   // initialization routines
   void init(const int n, const libbase::matrix<int>& codebook,
         const libcomm::bsid& chan)
      {
      this->n = n;
      this->codebook = codebook;
      computer = chan.get_computer();
#if DEBUG>=2
      std::cerr << "Initialize dminner2 computer..." << std::endl;
      std::cerr << "n = " << this->n << std::endl;
      std::cerr << "codebook = " << libbase::vector<int>(this->codebook) << std::endl;
      std::cerr << "sizeof(codebook) = " << sizeof(this->codebook) << std::endl;
      std::cerr << "N = " << computer.N << std::endl;
      std::cerr << "I = " << computer.I << std::endl;
      std::cerr << "xmax = " << computer.xmax << std::endl;
      std::cerr << "Rval = " << computer.Rval << std::endl;
      std::cerr << "Rtable = " << libbase::matrix<libcomm::bsid::real>(
            computer.Rtable) << std::endl;
#endif
      }
   void init(const libbase::vector<int>& marker) const
      {
      this->marker = marker;
#if DEBUG>=2
      std::cerr << "Initialize dminner2 computer..." << std::endl;
      std::cerr << "marker = " << libbase::vector<int>(this->marker) << std::endl;
      std::cerr << "sizeof(marker) = " << sizeof(this->marker) << std::endl;
#endif
      }
#ifdef __CUDACC__
   // batch receiver interface
   __device__
   void R(int d, int i, const cuda::vector_reference<bool>& r,
         cuda::vector_reference<libcomm::bsid::real>& ptable) const
      {
      const int w = marker(i); // marker vector
      const int s = codebook(i % codebook.get_rows(), d);
      // 'tx' is the vector of transmitted symbols that we're considering
      // TODO: find a way to use dminner::encode()
      cuda::bitfield tx(w ^ s, n);
      // compute the conditional probabilities
      computer.receive(tx, r, ptable);
      }
#endif
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
