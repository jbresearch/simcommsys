/*!
 * \file
 * 
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "embedder.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// *** Common Data Embedder/Extractor Interface ***

// Explicit Realizations

template class basic_embedder<int> ;
template class basic_embedder<float> ;
template class basic_embedder<double> ;

} // end namespace
