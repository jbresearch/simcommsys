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
