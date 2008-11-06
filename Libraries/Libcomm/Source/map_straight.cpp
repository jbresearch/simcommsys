/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_straight.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Serialization Support

const libbase::serializer map_straight::shelper("mapper", "map_straight", map_straight::create);

// Interface with mapper

/*! \copydoc mapper::setup()

   \note Each encoder output must be represented by an integral number of
         modulation symbols
*/
void map_straight::setup()
   {
   s1 = get_rate(M, N);
   s2 = get_rate(M, S);
   upsilon = tau*s1/s2;
   assertalways(tau*s1 == upsilon*s2);
   }

void map_straight::dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   assertalways(in.size() == tau);
   // Initialize results vector
   out.init(tau*s1);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = in(t); i<s1; i++, k++, x /= M)
         out(k) = x % M;
   }

void map_straight::doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const
   {
   assertalways(pin.size() == tau*s1);
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

std::string map_straight::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper";
   return sout.str();
   }

// Serialization Support

std::ostream& map_straight::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& map_straight::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
