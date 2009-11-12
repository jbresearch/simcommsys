/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "blockmodem.h"
#include "gf.h"
#include "logrealfast.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// *** Blockwise Modulator Common Interface ***

// Block modem operations

template <class S, template <class > class C, class dbl>
void basic_blockmodem<S, C, dbl>::modulate(const int N, const C<int>& encoded,
      C<S>& tx)
   {
   test_invariant();
   advance_always();
   domodulate(N, encoded, tx);
   }

template <class S, template <class > class C, class dbl>
void basic_blockmodem<S, C, dbl>::demodulate(const channel<S, C>& chan,
      const C<S>& rx, C<array1d_t>& ptable)
   {
   test_invariant();
   advance_if_dirty();
   dodemodulate(chan, rx, ptable);
   mark_as_dirty();
   }

// Explicit Realizations

using libbase::vector;
using libbase::matrix;
using libbase::gf;
using libbase::logrealfast;

template class basic_blockmodem<gf<1, 0x3> , vector, double> ;
template class basic_blockmodem<gf<2, 0x7> , vector, double> ;
template class basic_blockmodem<gf<3, 0xB> , vector, double> ;
template class basic_blockmodem<gf<4, 0x13> , vector, double> ;
template class basic_blockmodem<gf<5, 0x25> , vector, double> ;
template class basic_blockmodem<gf<6, 0x43> , vector, double> ;
template class basic_blockmodem<gf<7, 0x89> , vector, double> ;
template class basic_blockmodem<gf<8, 0x11D> , vector, double> ;
template class basic_blockmodem<gf<9, 0x211> , vector, double> ;
template class basic_blockmodem<gf<10, 0x409> , vector, double> ;
template class basic_blockmodem<bool, vector, double> ;
template class basic_blockmodem<sigspace, vector, double> ;

template class basic_blockmodem<gf<1, 0x3> , matrix, double> ;
template class basic_blockmodem<gf<2, 0x7> , matrix, double> ;
template class basic_blockmodem<gf<3, 0xB> , matrix, double> ;
template class basic_blockmodem<gf<4, 0x13> , matrix, double> ;
template class basic_blockmodem<gf<5, 0x25> , matrix, double> ;
template class basic_blockmodem<gf<6, 0x43> , matrix, double> ;
template class basic_blockmodem<gf<7, 0x89> , matrix, double> ;
template class basic_blockmodem<gf<8, 0x11D> , matrix, double> ;
template class basic_blockmodem<gf<9, 0x211> , matrix, double> ;
template class basic_blockmodem<gf<10, 0x409> , matrix, double> ;
template class basic_blockmodem<bool, matrix, double> ;
template class basic_blockmodem<sigspace, matrix, double> ;

template class basic_blockmodem<gf<1, 0x3> , vector, logrealfast> ;
template class basic_blockmodem<gf<2, 0x7> , vector, logrealfast> ;
template class basic_blockmodem<gf<3, 0xB> , vector, logrealfast> ;
template class basic_blockmodem<gf<4, 0x13> , vector, logrealfast> ;
template class basic_blockmodem<gf<5, 0x25> , vector, logrealfast> ;
template class basic_blockmodem<gf<6, 0x43> , vector, logrealfast> ;
template class basic_blockmodem<gf<7, 0x89> , vector, logrealfast> ;
template class basic_blockmodem<gf<8, 0x11D> , vector, logrealfast> ;
template class basic_blockmodem<gf<9, 0x211> , vector, logrealfast> ;
template class basic_blockmodem<gf<10, 0x409> , vector, logrealfast> ;
template class basic_blockmodem<bool, vector, logrealfast> ;
template class basic_blockmodem<sigspace, vector, logrealfast> ;

template class basic_blockmodem<gf<1, 0x3> , matrix, logrealfast> ;
template class basic_blockmodem<gf<2, 0x7> , matrix, logrealfast> ;
template class basic_blockmodem<gf<3, 0xB> , matrix, logrealfast> ;
template class basic_blockmodem<gf<4, 0x13> , matrix, logrealfast> ;
template class basic_blockmodem<gf<5, 0x25> , matrix, logrealfast> ;
template class basic_blockmodem<gf<6, 0x43> , matrix, logrealfast> ;
template class basic_blockmodem<gf<7, 0x89> , matrix, logrealfast> ;
template class basic_blockmodem<gf<8, 0x11D> , matrix, logrealfast> ;
template class basic_blockmodem<gf<9, 0x211> , matrix, logrealfast> ;
template class basic_blockmodem<gf<10, 0x409> , matrix, logrealfast> ;
template class basic_blockmodem<bool, matrix, logrealfast> ;
template class basic_blockmodem<sigspace, matrix, logrealfast> ;

// *** Blockwise Modulator Common Interface ***

template class blockmodem<gf<1, 0x3> , vector, double> ;
template class blockmodem<gf<2, 0x7> , vector, double> ;
template class blockmodem<gf<3, 0xB> , vector, double> ;
template class blockmodem<gf<4, 0x13> , vector, double> ;
template class blockmodem<gf<5, 0x25> , vector, double> ;
template class blockmodem<gf<6, 0x43> , vector, double> ;
template class blockmodem<gf<7, 0x89> , vector, double> ;
template class blockmodem<gf<8, 0x11D> , vector, double> ;
template class blockmodem<gf<9, 0x211> , vector, double> ;
template class blockmodem<gf<10, 0x409> , vector, double> ;
template class blockmodem<bool, vector, double> ;
template class blockmodem<sigspace, vector, double> ;

template class blockmodem<gf<1, 0x3> , matrix, double> ;
template class blockmodem<gf<2, 0x7> , matrix, double> ;
template class blockmodem<gf<3, 0xB> , matrix, double> ;
template class blockmodem<gf<4, 0x13> , matrix, double> ;
template class blockmodem<gf<5, 0x25> , matrix, double> ;
template class blockmodem<gf<6, 0x43> , matrix, double> ;
template class blockmodem<gf<7, 0x89> , matrix, double> ;
template class blockmodem<gf<8, 0x11D> , matrix, double> ;
template class blockmodem<gf<9, 0x211> , matrix, double> ;
template class blockmodem<gf<10, 0x409> , matrix, double> ;
template class blockmodem<bool, matrix, double> ;
template class blockmodem<sigspace, matrix, double> ;

template class blockmodem<gf<1, 0x3> , vector, logrealfast> ;
template class blockmodem<gf<2, 0x7> , vector, logrealfast> ;
template class blockmodem<gf<3, 0xB> , vector, logrealfast> ;
template class blockmodem<gf<4, 0x13> , vector, logrealfast> ;
template class blockmodem<gf<5, 0x25> , vector, logrealfast> ;
template class blockmodem<gf<6, 0x43> , vector, logrealfast> ;
template class blockmodem<gf<7, 0x89> , vector, logrealfast> ;
template class blockmodem<gf<8, 0x11D> , vector, logrealfast> ;
template class blockmodem<gf<9, 0x211> , vector, logrealfast> ;
template class blockmodem<gf<10, 0x409> , vector, logrealfast> ;
template class blockmodem<bool, vector, logrealfast> ;
template class blockmodem<sigspace, vector, logrealfast> ;

template class blockmodem<gf<1, 0x3> , matrix, logrealfast> ;
template class blockmodem<gf<2, 0x7> , matrix, logrealfast> ;
template class blockmodem<gf<3, 0xB> , matrix, logrealfast> ;
template class blockmodem<gf<4, 0x13> , matrix, logrealfast> ;
template class blockmodem<gf<5, 0x25> , matrix, logrealfast> ;
template class blockmodem<gf<6, 0x43> , matrix, logrealfast> ;
template class blockmodem<gf<7, 0x89> , matrix, logrealfast> ;
template class blockmodem<gf<8, 0x11D> , matrix, logrealfast> ;
template class blockmodem<gf<9, 0x211> , matrix, logrealfast> ;
template class blockmodem<gf<10, 0x409> , matrix, logrealfast> ;
template class blockmodem<bool, matrix, logrealfast> ;
template class blockmodem<sigspace, matrix, logrealfast> ;

} // end namespace
