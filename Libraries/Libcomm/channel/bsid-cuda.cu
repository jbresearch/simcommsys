/*!
 * \file
 * \brief   Parallel code for BSID channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
