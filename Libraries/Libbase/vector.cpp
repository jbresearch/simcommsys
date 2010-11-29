/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "vector.h"

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of memory allocation/deallocation
// 3 - Trace memory allocation/deallocation
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

#if DEBUG>=2
std::map<const void*,int> _vector_heap;
std::map<std::pair<const void*, int>, int> _vector_refs;
#endif

} // end namespace
