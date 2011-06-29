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

#include "map_stipple.h"
#include "vectorutils.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template <class > class C, class dbl>
void map_stipple<C, dbl>::advance() const
   {
   assertalways(size > 0);
   assertalways(sets > 0);
   // check if matrix is already set
   if (pattern.size() == size.length())
      return;
   // compute required sizes
   const int tau = size.length() / (sets + 1);
   assertalways(size.length() == tau * (sets + 1));
   // initialise the pattern matrix
   pattern.init(size.length());
   for (int i = 0, t = 0; t < tau; t++)
      for (int s = 0; s <= sets; s++, i++)
         pattern(i) = (s == 0 || (s - 1) == t % sets);
   }

template <template <class > class C, class dbl>
void map_stipple<C, dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   C<int> s;
   Base::dotransform(in, s);
   // final vector size depends on the number of set positions
   assertalways(s.size()==pattern.size());
   out.init(This::output_block_size());
   // puncture the results
   for (int i = 0, ii = 0; i < s.size(); i++)
      if (pattern(i))
         out(ii++) = s(i);
   }

template <template <class > class C, class dbl>
void map_stipple<C, dbl>::doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   assertalways(pin.size() == This::output_block_size());
   assertalways(pin(0).size() == M);
   // final matrix size depends on the number of set positions
   C<array1d_t> ptable;
   libbase::allocate(ptable, pattern.size(), M);
   // invert the puncturing
   for (int i = 0, ii = 0; i < pattern.size(); i++)
      if (pattern(i))
         {
         for (int j = 0; j < M; j++)
            ptable(i)(j) = pin(ii)(j);
         ii++;
         }
      else
         {
         for (int j = 0; j < M; j++)
            ptable(i)(j) = 1.0 / M;
         }
   // do the base (straight) inverse mapping
   Base::doinverse(ptable, pout);
   }

// Description

template <template <class > class C, class dbl>
std::string map_stipple<C, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Stipple Mapper (" << sets << ")";
   return sout.str();
   }

// Serialization Support

template <template <class > class C, class dbl>
std::ostream& map_stipple<C, dbl>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   sout << "# Stipple stride" << std::endl;
   sout << sets << std::endl;
   return sout;
   }

template <template <class > class C, class dbl>
std::istream& map_stipple<C, dbl>::serialize(std::istream& sin)
   {
   Base::serialize(sin);
   sin >> libbase::eatcomments >> sets;
   return sin;
   }

// Explicit instantiations

template class map_stipple<libbase::vector> ;
template <>
const libbase::serializer map_stipple<libbase::vector>::shelper("mapper",
      "map_stipple<vector>", map_stipple<libbase::vector>::create);

} // end namespace
