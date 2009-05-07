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
template <template<class> class C, class dbl>
void map_straight<C,dbl>::setup()
   {
   s1 = get_rate(M, N);
   s2 = get_rate(M, S);
   upsilon = size.x*s1/s2;
   assertalways(size.x*s1 == upsilon*s2);
   }

template <template<class> class C, class dbl>
void map_straight<C,dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results vector
   out.init(This::output_block_size());
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<size.x; t++)
      for(int i=0, x = in(t); i<s1; i++, k++, x /= M)
         out(k) = x % M;
   }

template <template<class> class C, class dbl>
void map_straight<C,dbl>::doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   assertalways(pin.size() == This::output_block_size());
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

template <template<class> class C, class dbl>
std::string map_straight<C,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper";
   return sout.str();
   }

// Serialization Support

template <template<class> class C, class dbl>
std::ostream& map_straight<C,dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <template<class> class C, class dbl>
std::istream& map_straight<C,dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::serializer;
using libbase::vector;
using libbase::logrealfast;

template class map_straight<vector>;
template <>
const serializer map_straight<vector>::shelper("mapper", "map_straight<vector>", map_straight<vector>::create);

template class map_straight<vector,float>;
template <>
const serializer map_straight<vector,float>::shelper("mapper", "map_straight<vector,float>", map_straight<vector,float>::create);

template class map_straight<vector,logrealfast>;
template <>
const serializer map_straight<vector,logrealfast>::shelper("mapper", "map_straight<vector,logrealfast>", map_straight<vector,logrealfast>::create);

}; // end namespace
