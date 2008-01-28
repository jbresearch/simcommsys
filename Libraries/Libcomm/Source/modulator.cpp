/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "modulator.h"
#include "serializer.h"
#include <stdlib.h>

namespace libcomm {

const libbase::serializer modulator<bool>::shelper("modulator", "modulator<bool>", modulator<bool>::create);

// Vector modem operations

void modulator<bool>::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx)
   {
   // Compute factors / sizes & check validity
   const int tau = encoded.size();
   const int s = int(round(log2(double(N))));
   if(N != (1<<s))
      {
      std::cerr << "FATAL ERROR (mapper): each encoder output (" << N << ") must be";
      std::cerr << " represented by an integral number of bits.";
      std::cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x>>=1)
         tx(k) = (x & 1);
   }

void modulator<bool>::demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   // Create a matrix of all possible transmitted symbols
   libbase::vector<bool> tx(2);
   tx(0) = false;
   tx(1) = true;
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description & Serialization

std::string modulator<bool>::description() const
   {
   return "Binary Modulation";
   }

}; // end namespace
