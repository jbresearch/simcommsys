/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mapper.h"
#include "gf.h"
#include "serializer.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Serialization Support

const libbase::serializer mapper::shelper("mapper", "straight", mapper::create);

// Vector mapper operations

void mapper::transform(const int N, const libbase::vector<int>& encoded, const int M, libbase::vector<int>& tx)
   {
   //// Compute factors / sizes & check validity
   //const int M = num_symbols();
   //const int tau = encoded.size();
   //const int s = int(round( log2(double(N)) / log2(double(M)) ));
   //if(N != pow(num_symbols(),s))
   //   {
   //   std::cerr << "FATAL ERROR (mapper): each encoder output (" << N << ") must be";
   //   std::cerr << " represented by an integral number of modulation symbols (" << M << ").";
   //   std::cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
   //   exit(1);
   //   }
   //// Initialize results vector
   //tx.init(tau*s);
   //// Modulate encoded stream (least-significant first)
   //for(int t=0, k=0; t<tau; t++)
   //   for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
   //      tx(k) = modulate(x % M);
   }

void mapper::inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout)
   {
   //// Compute sizes
   //const int M = num_symbols();
   //// Create a matrix of all possible transmitted symbols
   //libbase::vector tx(M);
   //for(int x=0; x<M; x++)
   //   tx(x) = modulate(x);
   //// Work out the probabilities of each possible signal
   //chan.receive(tx, rx, ptable);
   }

// Description & Serialization

std::string mapper::description() const
   {
   std::ostringstream sout;
   sout << "Direct Mapper";
   return sout.str();
   }


}; // end namespace
