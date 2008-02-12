/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "modulator.h"
#include "gf.h"
#include "serializer.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// *** Templated GF(q) modulator ***

// Vector modem operations

template <class G> void modulator<G>::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx)
   {
   // Compute factors / sizes & check validity
   const int tau = encoded.size();
   const int s = int(round( log2(double(N)) / log2(double(num_symbols())) ));
   if(N != (1<<s))
      {
      std::cerr << "FATAL ERROR (mapper): each encoder output (" << N << ") must be";
      std::cerr << " represented by an integral number of symbols.";
      std::cerr << " Suggested number of mod. symbols/encoder output was " << s << ".\n";
      exit(1);
      }
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x>>=1)
         tx(k) = (x & (num_symbols()-1));
   }

template <class G> void modulator<G>::demodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::matrix<double>& ptable)
   {
   // Create a matrix of all possible transmitted symbols
   libbase::vector<G> tx(num_symbols());
   for(int i=0; i<num_symbols(); i++)
      tx(i) = G(i);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description & Serialization

template <class G> std::string modulator<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << num_symbols() << ") Modulation";
   return sout.str();
   }

// Explicit Realizations

template class modulator< libbase::gf<1,0x3> >;
template <> const libbase::serializer modulator< libbase::gf<1,0x3> >::shelper("modulator", "modulator<gf<1,0x3>>", modulator< libbase::gf<1,0x3> >::create);
template class modulator< libbase::gf<2,0x7> >;
template <> const libbase::serializer modulator< libbase::gf<2,0x7> >::shelper("modulator", "modulator<gf<2,0x7>>", modulator< libbase::gf<2,0x7> >::create);
template class modulator< libbase::gf<3,0xB> >;
template <> const libbase::serializer modulator< libbase::gf<3,0xB> >::shelper("modulator", "modulator<gf<3,0xB>>", modulator< libbase::gf<3,0xB> >::create);
template class modulator< libbase::gf<4,0x13> >;
template <> const libbase::serializer modulator< libbase::gf<4,0x13> >::shelper("modulator", "modulator<gf<4,0x13>>", modulator< libbase::gf<4,0x13> >::create);


// *** Specific to modulator<bool> ***

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
