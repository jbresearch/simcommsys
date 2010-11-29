/*!
 * \file
 * \brief   CUDA value in device memory.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "cuda-all.h"

namespace cuda {

// explicit instantiations

template class value<bool> ;
template class value<int> ;
template class value<float> ;
template class value<double> ;

} // end namespace
