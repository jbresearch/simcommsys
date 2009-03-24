/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_interleaved.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template<class> class C>
void map_interleaved<C>::advance() const
   {
   lut.init(this->output_block_size(),r);
   }

template <template<class> class C>
void map_interleaved<C>::dotransform(const C<int>& in, C<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   C<int> s;
   map_straight<C>::dotransform(in, s);
   // final vector is the same size as straight-mapped one
   out.init(s);
   // shuffle the results
   assert(out.size() == lut.size());
   for(int i=0; i<out.size(); i++)
      out(lut(i)) = s(i);
   }

template <template<class> class C>
void map_interleaved<C>::doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   assert(pin.size() == lut.size());
   // temporary matrix is the same size as input
   C<array1d_t> ptable;
   ptable.init(lut.size());
   // invert the shuffling
   for(int i=0; i<lut.size(); i++)
      ptable(i) = pin(lut(i));
   // do the base (straight) mapping
   map_straight<C>::doinverse(ptable, pout);
   }

// Description

template <template<class> class C>
std::string map_interleaved<C>::description() const
   {
   std::ostringstream sout;
   sout << "Interleaved Mapper";
   return sout.str();
   }

// Serialization Support

template <template<class> class C>
std::ostream& map_interleaved<C>::serialize(std::ostream& sout) const
   {
   map_straight<C>::serialize(sout);
   return sout;
   }

template <template<class> class C>
std::istream& map_interleaved<C>::serialize(std::istream& sin)
   {
   map_straight<C>::serialize(sin);
   return sin;
   }

// Explicit instantiations

template class map_interleaved<libbase::vector>;
template <>
const libbase::serializer map_interleaved<libbase::vector>::shelper("mapper", "map_interleaved<vector>", map_interleaved<libbase::vector>::create);

}; // end namespace
