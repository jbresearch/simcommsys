/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_stipple.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Interface with mapper

template <template<class> class C, class dbl>
void map_stipple<C,dbl>::advance() const
   {
   assertalways(size > 0);
   assertalways(sets > 0);
   // check if matrix is already set
   if(pattern.size() == size.length()*(sets+1))
      return;
   // initialise the pattern matrix
   pattern.init(size.length()*(sets+1));
   for(int i=0, t=0; t<size.length(); t++)
      for(int s=0; s<=sets; s++, i++)
         pattern(i) = (s==0 || (s-1)==t%sets);
   }

template <template<class> class C, class dbl>
void map_stipple<C,dbl>::dotransform(const C<int>& in, C<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   C<int> s;
   Base::dotransform(in, s);
   // final vector size depends on the number of set positions
   assertalways(s.size()==pattern.size());
   out.init(This::output_block_size());
   // puncture the results
   for(int i=0, ii=0; i<s.size(); i++)
      if(pattern(i))
         out(ii++) = s(i);
   }

template <template<class> class C, class dbl>
void map_stipple<C,dbl>::doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const
   {
   assertalways(pin.size() == This::output_block_size());
   assertalways(pin(0).size() == M);
   // final matrix size depends on the number of set positions
   C<array1d_t> ptable;
   ptable.init(pattern.size());
   for(int i=0; i<pattern.size(); i++)
      ptable(i).init(M);
   // invert the puncturing
   for(int i=0, ii=0; i<pattern.size(); i++)
      if(pattern(i))
         {
         for(int j=0; j<M; j++)
            ptable(i)(j) = pin(ii)(j);
         ii++;
         }
      else
         {
         for(int j=0; j<M; j++)
            ptable(i)(j) = 1.0/M;
         }
   // do the base (straight) inverse mapping
   Base::doinverse(ptable, pout);
   }

// Description

template <template<class> class C, class dbl>
std::string map_stipple<C,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Stipple Mapper (" << sets << ")";
   return sout.str();
   }

// Serialization Support

template <template<class> class C, class dbl>
std::ostream& map_stipple<C,dbl>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   sout << sets << "\n";
   return sout;
   }

template <template<class> class C, class dbl>
std::istream& map_stipple<C,dbl>::serialize(std::istream& sin)
   {
   Base::serialize(sin);
   sin >> sets;
   return sin;
   }

// Explicit instantiations

template class map_stipple<libbase::vector>;
template <>
const libbase::serializer map_stipple<libbase::vector>::shelper("mapper", "map_stipple<vector>", map_stipple<libbase::vector>::create);

}; // end namespace
