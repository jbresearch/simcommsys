/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "map_permuted.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template <class > class C, class dbl>
void map_permuted<C, dbl>::advance() const
   {
   lut.init(This::output_block_size());
   for (int i = 0; i < This::output_block_size(); i++)
      lut(i).init(M, r);
   }

template <template <class > class C, class dbl>
void map_permuted<C, dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   C<int> s;
   Base::dotransform(in, s);
   // final vector is the same size as straight-mapped one
   out.init(s.size());
   // permute the results
   assert(out.size() == lut.size());
   for (int i = 0; i < out.size(); i++)
      out(i) = lut(i)(s(i));
   }

template <template <class > class C, class dbl>
void map_permuted<C, dbl>::doinverse(const C<array1d_t>& pin,
      C<array1d_t>& pout) const
   {
   assert(pin.size() == lut.size());
   assert(pin(0).size() == M);
   // temporary matrix is the same size as input
   C<array1d_t> ptable;
   ptable.init(lut.size());
   for (int i = 0; i < lut.size(); i++)
      ptable(i).init(M);
   // invert the permutation
   for (int i = 0; i < lut.size(); i++)
      for (int j = 0; j < M; j++)
         ptable(i)(j) = pin(i)(lut(i)(j));
   // do the base (straight) mapping
   Base::doinverse(ptable, pout);
   }

// Description

template <template <class > class C, class dbl>
std::string map_permuted<C, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Permuted Mapper";
   return sout.str();
   }

// Serialization Support

template <template <class > class C, class dbl>
std::ostream& map_permuted<C, dbl>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   return sout;
   }

template <template <class > class C, class dbl>
std::istream& map_permuted<C, dbl>::serialize(std::istream& sin)
   {
   Base::serialize(sin);
   return sin;
   }

// Explicit instantiations

template class map_permuted<libbase::vector>
template <>
const libbase::serializer map_permuted<libbase::vector>::shelper("mapper",
      "map_permuted<vector>", map_permuted<libbase::vector>::create);

} // end namespace
