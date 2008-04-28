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

// Vector map_straight operations

void map_straight::transform(const int N, const libbase::vector<int>& encoded, const int M, libbase::vector<int>& tx)
   {
   // Compute factors / sizes & check validity
   const int tau = encoded.size();
   const int s = int(round( log2(double(N)) / log2(double(M)) ));
   // Each encoder output must be represented by an integral number of modulation symbols
   assertalways(N == pow(M,s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = x % M;
   }

void map_straight::inverse(const libbase::matrix<double>& pin, const int N, libbase::matrix<double>& pout)
   {
   // Compute factors / sizes & check validity
   const int M = pin.ysize();
   const int s = int(round(log(double(N))/log(double(M))));
   // Each encoder symbol must be represented by an integral number of channel symbols
   assertalways(N == pow(M,s));
   // Determine required length of encoder sequence
   const int tau = pin.xsize() / s;
   // Confirm channel sequence to be of the correct length
   assertalways(pin.xsize() == tau*s);
   // Initialize results vector
   pout.init(tau,N);
   // Get the necessary data from the channel
   for(int t=0; t<tau; t++)
      for(int x=0; x<N; x++)
         {
         pout(t,x) = 1;
         for(int i=0, thisx = x; i<s; i++, thisx /= M)
            pout(t, x) *= pin(t*s+i, thisx % M);
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
