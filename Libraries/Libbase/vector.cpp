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

#if DEBUG>=2
std::map<const void*,int> _vector_heap;
std::map<std::pair<const void*, int>, int> _vector_refs;
#endif

} // end namespace
