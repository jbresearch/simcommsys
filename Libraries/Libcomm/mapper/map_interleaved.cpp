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

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "map_interleaved.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template <class > class C, class dbl>
void map_interleaved<C, dbl>::advance() const
   {
   lut.init(This::output_block_size(), r);
   }

template <template <class > class C, class dbl>
void map_interleaved<C, dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   C<int> s;
   Base::dotransform(in, s);
   // final vector is the same size as straight-mapped one
   out.init(s.size());
   // shuffle the results
   assert(out.size() == lut.size());
   for (int i = 0; i < out.size(); i++)
      out(lut(i)) = s(i);
   }

template <template <class > class C, class dbl>
void map_interleaved<C, dbl>::doinverse(const C<array1d_t>& pin,
      C<array1d_t>& pout) const
   {
   assert(pin.size() == lut.size());
   // temporary matrix is the same size as input
   C<array1d_t> ptable;
   ptable.init(lut.size());
   // invert the shuffling
   for (int i = 0; i < lut.size(); i++)
      ptable(i) = pin(lut(i));
   // do the base (straight) mapping
   Base::doinverse(ptable, pout);
   }

// Description

template <template <class > class C, class dbl>
std::string map_interleaved<C, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Interleaved Mapper";
   return sout.str();
   }

// Serialization Support

template <template <class > class C, class dbl>
std::ostream& map_interleaved<C, dbl>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   return sout;
   }

template <template <class > class C, class dbl>
std::istream& map_interleaved<C, dbl>::serialize(std::istream& sin)
   {
   Base::serialize(sin);
   return sin;
   }

// Explicit instantiations

template class map_interleaved<libbase::vector> ;
template <>
const libbase::serializer map_interleaved<libbase::vector>::shelper("mapper",
      "map_interleaved<vector>", map_interleaved<libbase::vector>::create);

} // end namespace
