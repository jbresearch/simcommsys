/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_straight.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Interface with mapper

/*! \copydoc mapper::setup()

   \note Each encoder output must be represented by an integral number of
         modulation symbols
*/
template <template<class> class C>
void map_straight<C>::setup()
   {
   s1 = get_rate(M, N);
   s2 = get_rate(M, S);
   upsilon = size.x*s1/s2;
   assertalways(size.x*s1 == upsilon*s2);
   }

template <template<class> class C>
void map_straight<C>::dotransform(const C<int>& in, C<int>& out) const
   {
   assertalways(in.size() == map_straight<C>::input_block_size());
   // Initialize results vector
   out.init(map_straight<C>::output_block_size());
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<size.x; t++)
      for(int i=0, x = in(t); i<s1; i++, k++, x /= M)
         out(k) = x % M;
   }

template <template<class> class C>
void map_straight<C>::doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   assertalways(pin.size() == map_straight<C>::output_block_size());
   // Initialize results vector
   pout.init(upsilon);
   for(int t=0; t<upsilon; t++)
      {
      assertalways(pin(t).size() == M);
      pout(t).init(S);
      }
   // Get the necessary data from the channel
   for(int t=0; t<upsilon; t++)
      for(int x=0; x<S; x++)
         {
         pout(t)(x) = 1;
         for(int i=0, thisx = x; i<s2; i++, thisx /= M)
            pout(t)(x) *= pin(t*s2+i)(thisx % M);
         }
   }

// Description

template <template<class> class C>
std::string map_straight<C>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper";
   return sout.str();
   }

// Serialization Support

template <template<class> class C>
std::ostream& map_straight<C>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <template<class> class C>
std::istream& map_straight<C>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit instantiations

template class map_straight<libbase::vector>;
template <>
const libbase::serializer map_straight<libbase::vector>::shelper("mapper", "map_straight<vector>", map_straight<libbase::vector>::create);

}; // end namespace
