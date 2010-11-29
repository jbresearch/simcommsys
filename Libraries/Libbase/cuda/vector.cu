/*!
 * \file
 * \brief   CUDA vector in device memory.
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

template class vector<bool> ;
template class vector<int> ;
template class vector<float> ;
template class vector<double> ;

} // end namespace
