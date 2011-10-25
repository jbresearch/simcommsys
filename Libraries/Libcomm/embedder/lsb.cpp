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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "lsb.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

using libbase::serializer;

// description output

template <class S>
std::string lsb<S>::description() const
   {
   std::ostringstream sout;
   sout << "LSB Replacement embedder (M=" << M << ")";
   return sout.str();
   }

// object serialization - saving

template <class S>
std::ostream& lsb<S>::serialize(std::ostream& sout) const
   {
   sout << M << std::endl;
   return sout;
   }

// object serialization - loading

template <class S>
std::istream& lsb<S>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> M;
   return sin;
   }

// Explicit Realizations

template class lsb<int>;
template <>
const serializer lsb<int>::shelper("embedder", "lsb<int>", lsb<int>::create);

} // end namespace
