/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "blockmodem.h"
#include "gf.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// *** Blockwise Modulator Common Interface ***

// Vector modem operations

template <class S, template<class> class C>
void basic_blockmodem<S,C>::modulate(const int N, const C<int>& encoded, C<S>& tx)
   {
   test_invariant();
   advance_always();
   domodulate(N, encoded, tx);
   }

template <class S, template<class> class C>
void basic_blockmodem<S,C>::demodulate(const channel<S,C>& chan, const C<S>& rx, C<array1d_t>& ptable)
   {
   test_invariant();
   advance_if_dirty();
   dodemodulate(chan, rx, ptable);
   mark_as_dirty();
   }

// Explicit Realizations

using libbase::matrix;
using libbase::gf;
using libcomm::sigspace;

template class basic_blockmodem< gf<1,0x3> >;
template class basic_blockmodem< gf<2,0x7> >;
template class basic_blockmodem< gf<3,0xB> >;
template class basic_blockmodem< gf<4,0x13> >;
template class basic_blockmodem<bool>;
template class basic_blockmodem<sigspace>;

template class basic_blockmodem< gf<1,0x3>,matrix >;
template class basic_blockmodem< gf<2,0x7>,matrix >;
template class basic_blockmodem< gf<3,0xB>,matrix >;
template class basic_blockmodem< gf<4,0x13>,matrix >;
template class basic_blockmodem<bool,matrix>;
template class basic_blockmodem<sigspace,matrix>;

// *** Blockwise Modulator Common Interface ***

// Explicit Realizations

template class blockmodem< gf<1,0x3> >;
template class blockmodem< gf<2,0x7> >;
template class blockmodem< gf<3,0xB> >;
template class blockmodem< gf<4,0x13> >;
template class blockmodem<bool>;
template class blockmodem<sigspace>;

template class blockmodem< gf<1,0x3>,matrix >;
template class blockmodem< gf<2,0x7>,matrix >;
template class blockmodem< gf<3,0xB>,matrix >;
template class blockmodem< gf<4,0x13>,matrix >;
template class blockmodem<bool,matrix>;
template class blockmodem<sigspace,matrix>;


// *** Templated GF(q) blockmodem ***

// Vector modem operations

template <class G>
void direct_blockmodem<G>::domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx)
   {
   // Inherit sizes
   const int M = this->num_symbols();
   const int tau = this->input_block_size();
   // Compute factors & check validity
   const int s = int(round( log2(double(N)) / log2(double(M)) ));
   assertalways(tau == encoded.size());
   // Each encoder output N must be representable by an integral number of
   // modulation symbols M
   assertalways(N == pow(M,s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x /= M)
         tx(k) = direct_modem_implementation<G>::modulate(x % M);
   }

template <class G>
void direct_blockmodem<G>::dodemodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::vector<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->num_symbols();
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == rx.size());
   // Create a matrix of all possible transmitted symbols
   libbase::vector<G> tx(M);
   for(int x=0; x<M; x++)
      tx(x) = direct_modem_implementation<G>::modulate(x);
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description

template <class G>
std::string direct_blockmodem<G>::description() const
   {
   std::ostringstream sout;
   sout << "Blockwise " << direct_modem_implementation<G>::description();
   return sout.str();
   }

// Serialization Support

template <class G>
std::ostream& direct_blockmodem<G>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class G>
std::istream& direct_blockmodem<G>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit Realizations

using libbase::serializer;
using libbase::gf;
using libbase::matrix;

template class direct_blockmodem< gf<1,0x3> >;
template <>
const serializer direct_blockmodem< gf<1,0x3> >::shelper("blockmodem", "blockmodem<gf<1,0x3>>", direct_blockmodem< gf<1,0x3> >::create);
template class direct_blockmodem< gf<2,0x7> >;
template <>
const serializer direct_blockmodem< gf<2,0x7> >::shelper("blockmodem", "blockmodem<gf<2,0x7>>", direct_blockmodem< gf<2,0x7> >::create);
template class direct_blockmodem< gf<3,0xB> >;
template <>
const serializer direct_blockmodem< gf<3,0xB> >::shelper("blockmodem", "blockmodem<gf<3,0xB>>", direct_blockmodem< gf<3,0xB> >::create);
template class direct_blockmodem< gf<4,0x13> >;
template <>
const serializer direct_blockmodem< gf<4,0x13> >::shelper("blockmodem", "blockmodem<gf<4,0x13>>", direct_blockmodem< gf<4,0x13> >::create);

const serializer direct_blockmodem<bool>::shelper("blockmodem", "blockmodem<bool>", direct_blockmodem<bool>::create);

// *** Specific to direct_blockmodem<bool> ***

// Vector modem operations

void direct_blockmodem<bool>::domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx)
   {
   // Inherit sizes
   const int tau = input_block_size();
   // Compute factors & check validity
   const int s = int(round( log2(double(N)) ));
   assertalways(tau == encoded.size());
   // Each encoder output N must be representable by an integral number of bits
   assertalways(N == (1<<s));
   // Initialize results vector
   tx.init(tau*s);
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<tau; t++)
      for(int i=0, x = encoded(t); i<s; i++, k++, x>>=1)
         tx(k) = (x & 1);
   }

void direct_blockmodem<bool>::dodemodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::vector<array1d_t>& ptable)
   {
   // Inherit sizes
   const int tau = input_block_size();
   // Check validity
   assertalways(tau == rx.size());
   // Create a matrix of all possible transmitted symbols
   libbase::vector<bool> tx(2);
   tx(0) = false;
   tx(1) = true;
   // Work out the probabilities of each possible signal
   chan.receive(tx, rx, ptable);
   }

// Description

std::string direct_blockmodem<bool>::description() const
   {
   std::ostringstream sout;
   sout << "Blockwise " << direct_modem_implementation<bool>::description();
   return sout.str();
   }

// Serialization Support

std::ostream& direct_blockmodem<bool>::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& direct_blockmodem<bool>::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
