/*!
 * \file
 * \brief   Parallel code for BPMR channel.
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

#include "bpmr.h"

namespace cuda {

// CUDA kernels

template <class real>
__global__
void receive_kernel(const typename libcomm::bpmr<real>::metric_computer object,
      const cuda::vector_reference<bool> tx,
      const cuda::vector_reference<bool> rx,
      cuda::vector_reference<real> ptable)
   {
   object.receive(tx, rx, ptable);
   }

} // end namespace

namespace libcomm {

template <class real>
void bpmr<real>::metric_computer::receive(const array1b_t& tx,
      const array1b_t& rx, array1r_t& ptable) const
   {
   // allocate space on device for result, and initialize
   cuda::vector<real> dev_ptable;
   dev_ptable.init(Zmax - Zmin + 1);
   // allocate space on device for tx and rx vectors, and copy over
   cuda::vector<bool> dev_tx;
   cuda::vector<bool> dev_rx;
   dev_tx = tx;
   dev_rx = rx;
   // call the kernel with a copy of this object
   cuda::receive_kernel<real> <<<1,1>>>(*this, dev_tx, dev_rx, dev_ptable);
   // return the result
   ptable = array1r_t(dev_ptable);
   }

} // end namespace

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>

#define REAL_TYPE_SEQ \
   (float)(double)

namespace cuda {

#define INSTANTIATE_FUNC(r, x, type) \
      template __global__ void receive_kernel( \
            const typename libcomm::bpmr<type>::metric_computer object, \
            const cuda::vector_reference<bool> tx, \
            const cuda::vector_reference<bool> rx, \
            cuda::vector_reference<type> ptable);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_FUNC, x, REAL_TYPE_SEQ)

} // end namespace

namespace libcomm {

#define INSTANTIATE_CLASS(r, x, type) \
      template class bpmr<type>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_CLASS, x, REAL_TYPE_SEQ)

} // end namespace
