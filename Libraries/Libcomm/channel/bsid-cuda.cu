/*!
 * \file
 * \brief   Parallel code for BSID channel.
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

#include "bsid.h"

namespace cuda {

__global__
void receive_kernel(const libcomm::bsid::metric_computer object,
      cuda::value_reference<libcomm::bsid::real> result, const cuda::bitfield tx,
      const cuda::vector_reference<bool> rx)
   {
   result() = object.receive(tx, rx);
   }

} // end namespace

namespace libcomm {

bsid::real bsid::metric_computer::receive(const bitfield& tx,
      const array1b_t& rx) const
   {
   // allocate space on device for result, and initialize
   cuda::value<real> dev_result;
   dev_result.init();
   // allocate space on device for rx vector, and copy over
   cuda::vector<bool> dev_rx;
   dev_rx = rx;
   // create device-compatible tx bitfield
   cuda::bitfield dev_tx(tx.size());
   dev_tx = tx;
   // call the kernel with a copy of this object
   cuda::receive_kernel<<<1,1>>>(*this, dev_result, dev_tx, dev_rx);
   cudaSafeWaitForKernel();
   // return the result
   return dev_result;
   }

} // end namespace
