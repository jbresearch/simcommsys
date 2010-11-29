/*!
 * \file
 * \brief   CUDA matrix in device memory.
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

template class matrix<int> ;
template class matrix<float> ;
template class matrix<double> ;

} // end namespace
