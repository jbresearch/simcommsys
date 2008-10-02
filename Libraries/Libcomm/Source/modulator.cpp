/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "modulator.h"
#include "gf.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// *** Common Modulator Interface ***

// Vector modem operations

template <class S>
void basic_modulator<S>::modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<S>& tx)
   {
   advance_always();
   domodulate(N, encoded, tx);
   }

template <class S>
void basic_modulator<S>::demodulate(const channel<S>& chan, const libbase::vector<S>& rx, libbase::matrix<double>& ptable)
   {
   advance_if_dirty();
   dodemodulate(chan, rx, ptable);
   mark_as_dirty();
   }

// Explicit Realizations

template class basic_modulator< libbase::gf<1,0x3> >;
template class basic_modulator< libbase::gf<2,0x7> >;
template class basic_modulator< libbase::gf<3,0xB> >;
template class basic_modulator< libbase::gf<4,0x13> >;
template class basic_modulator<bool>;
template class basic_modulator<libcomm::sigspace>;


// *** Templated GF(q) modulator ***

// Vector modem operations

template <class G>
void direct_modulator<G>::domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx)
   {
   // Compute factors / sizes & check validity
   const int M = num_symbols();
   const int tau = encoded.size();
   const int s = int(round( log2(double(N)) / log2(double(M)) ));
   // Each encoder output N must be representable by an integral number of
   // modulation symbols M
   assertalways(N == pow(num_symbols(),s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = modulate(x % M);
   }

template <class G>
void direct_modulator<G>::dodemodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::matrix<double>& ptable)
   {
   // Compute sizes
   const int M = num_symbols();
   // Create a matrix of all possible transmitted symbols
   libbase::vector<G> tx(M);
   for(int x=0; x<M; x++)
      tx(x) = modulate(x);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description

template <class G>
std::string direct_modulator<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << num_symbols() << ") Modulation";
   return sout.str();
   }

// Serialization Support

template <class G>
std::ostream& direct_modulator<G>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class G>
std::istream& direct_modulator<G>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit Realizations

template class direct_modulator< libbase::gf<1,0x3> >;
template <>
const libbase::serializer direct_modulator< libbase::gf<1,0x3> >::shelper("modulator", "modulator<gf<1,0x3>>", direct_modulator< libbase::gf<1,0x3> >::create);
template class direct_modulator< libbase::gf<2,0x7> >;
template <>
const libbase::serializer direct_modulator< libbase::gf<2,0x7> >::shelper("modulator", "modulator<gf<2,0x7>>", direct_modulator< libbase::gf<2,0x7> >::create);
template class direct_modulator< libbase::gf<3,0xB> >;
template <>
const libbase::serializer direct_modulator< libbase::gf<3,0xB> >::shelper("modulator", "modulator<gf<3,0xB>>", direct_modulator< libbase::gf<3,0xB> >::create);
template class direct_modulator< libbase::gf<4,0x13> >;
template <>
const libbase::serializer direct_modulator< libbase::gf<4,0x13> >::shelper("modulator", "modulator<gf<4,0x13>>", direct_modulator< libbase::gf<4,0x13> >::create);


// *** Specific to direct_modulator<bool> ***

const libbase::serializer direct_modulator<bool>::shelper("modulator", "modulator<bool>", direct_modulator<bool>::create);

// Vector modem operations

void direct_modulator<bool>::domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx)
   {
   // Compute factors / sizes & check validity
   const int tau = encoded.size();
   const int s = int(round(log2(double(N))));
   // Each encoder output N must be representable by an integral number of bits
   assertalways(N == (1<<s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x>>=1)
         tx(k) = (x & 1);
   }

void direct_modulator<bool>::dodemodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable)
   {
   // Create a matrix of all possible transmitted symbols
   libbase::vector<bool> tx(2);
   tx(0) = false;
   tx(1) = true;
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description

std::string direct_modulator<bool>::description() const
   {
   return "Binary Modulation";
   }

// Serialization Support

std::ostream& direct_modulator<bool>::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& direct_modulator<bool>::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
