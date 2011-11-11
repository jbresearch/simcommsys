/*!
 * \file
 * 
 * Copyright (c) 2010 Johann A. Briffa
 * 
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "direct_blockmodem.h"
#include "gf.h"
#include "logrealfast.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

using libbase::serializer;
using libbase::gf;
using libbase::logrealfast;
using libbase::vector;
using libbase::matrix;

// *** Vector GF(q) blockmodem ***

// Block modem operations

template <class G, class dbl>
void direct_blockmodem_implementation<G, vector, dbl>::domodulate(const int N,
      const vector<int>& encoded, vector<G>& tx)
   {
   // Inherit sizes
   const int tau = encoded.size();
   // Initialize results vector
   tx.init(tau);
   // Modulate encoded stream
   for (int t = 0; t < tau; t++)
      tx(t) = Implementation::modulate(encoded(t));
   }

template <class G, class dbl>
void direct_blockmodem_implementation<G, vector, dbl>::dodemodulate(
      const channel<G, vector>& chan, const vector<G>& rx,
      vector<array1d_t>& ptable)
   {
   // Inherit sizes
   const int M = this->num_symbols();
   // Allocate space for temporary results
   vector<vector<double> > ptable_double;
      {
      // Create a matrix of all possible transmitted symbols
      vector<G> tx(M);
      for (int x = 0; x < M; x++)
         tx(x) = Implementation::modulate(x);
      // Work out the probabilities of each possible signal
      chan.receive(tx, rx, ptable_double);
      }
   // Convert result
   ptable = ptable_double;
   }

// Explicit Realizations

template class direct_blockmodem_implementation<bool, vector, double> ;
template class direct_blockmodem_implementation<bool, vector, logrealfast> ;

// *** Templated GF(q) blockmodem ***

// Description

template <class G, template <class > class C, class dbl>
std::string direct_blockmodem<G, C, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Blockwise " << Implementation::description();
   return sout.str();
   }

// Serialization Support

template <class G, template <class > class C, class dbl>
std::ostream& direct_blockmodem<G, C, dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class G, template <class > class C, class dbl>
std::istream& direct_blockmodem<G, C, dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit Realizations

// Vector, double-precision

template class direct_blockmodem<gf<1, 0x3> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<1, 0x3> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<1,0x3>>", direct_blockmodem<gf<1, 0x3> ,
            vector, double>::create);

template class direct_blockmodem<gf<2, 0x7> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<2, 0x7> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<2,0x7>>", direct_blockmodem<gf<2, 0x7> ,
            vector, double>::create);

template class direct_blockmodem<gf<3, 0xB> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<3, 0xB> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<3,0xB>>", direct_blockmodem<gf<3, 0xB> ,
            vector, double>::create);

template class direct_blockmodem<gf<4, 0x13> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<4, 0x13> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<4,0x13>>", direct_blockmodem<gf<4, 0x13> ,
            vector, double>::create);

template class direct_blockmodem<gf<5, 0x25> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<5, 0x25> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<5,0x25>>", direct_blockmodem<gf<5,
            0x25> , vector, double>::create);

template class direct_blockmodem<gf<6, 0x43> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<6, 0x43> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<6,0x43>>", direct_blockmodem<gf<6,
            0x43> , vector, double>::create);

template class direct_blockmodem<gf<7, 0x89> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<7, 0x89> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<7,0x89>>", direct_blockmodem<gf<7,
            0x89> , vector, double>::create);

template class direct_blockmodem<gf<8, 0x11D> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<8, 0x11D> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<8,0x11D>>", direct_blockmodem<gf<8,
            0x11D> , vector, double>::create);

template class direct_blockmodem<gf<9, 0x211> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<9, 0x211> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<9,0x211>>", direct_blockmodem<gf<9,
            0x211> , vector, double>::create);

template class direct_blockmodem<gf<10, 0x409> , vector, double> ;
template <>
const serializer direct_blockmodem<gf<10, 0x409> , vector, double>::shelper(
      "blockmodem", "blockmodem<gf<10,0x409>>", direct_blockmodem<gf<10,
            0x409> , vector, double>::create);

template class direct_blockmodem<bool, vector, double> ;
template <>
const serializer direct_blockmodem<bool, vector, double>::shelper("blockmodem",
      "blockmodem<bool>", direct_blockmodem<bool, vector, double>::create);

// Vector, logrealfast-precision

template class direct_blockmodem<gf<1, 0x3> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<1, 0x3> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<1,0x3>,logrealfast>", direct_blockmodem<gf<
            1, 0x3> , vector, logrealfast>::create);

template class direct_blockmodem<gf<2, 0x7> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<2, 0x7> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<2,0x7>,logrealfast>", direct_blockmodem<gf<
            2, 0x7> , vector, logrealfast>::create);

template class direct_blockmodem<gf<3, 0xB> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<3, 0xB> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<3,0xB>,logrealfast>", direct_blockmodem<gf<
            3, 0xB> , vector, logrealfast>::create);

template class direct_blockmodem<gf<4, 0x13> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<4, 0x13> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<4,0x13>,logrealfast>", direct_blockmodem<gf<
            4, 0x13> , vector, logrealfast>::create);

template class direct_blockmodem<gf<5, 0x25> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<5, 0x25> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<5,0x25>,logrealfast>",
      direct_blockmodem<gf<5, 0x25> , vector, logrealfast>::create);

template class direct_blockmodem<gf<6, 0x43> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<6, 0x43> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<6,0x43>,logrealfast>",
      direct_blockmodem<gf<6, 0x43> , vector, logrealfast>::create);

template class direct_blockmodem<gf<7, 0x89> , vector, logrealfast> ;
template <>
const serializer direct_blockmodem<gf<7, 0x89> , vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<gf<7,0x89>,logrealfast>",
      direct_blockmodem<gf<7, 0x89> , vector, logrealfast>::create);

template class direct_blockmodem<gf<8, 0x11D> , vector, logrealfast> ;
template <>
const serializer
      direct_blockmodem<gf<8, 0x11D> , vector, logrealfast>::shelper(
            "blockmodem", "blockmodem<gf<8,0x11D>,logrealfast>",
            direct_blockmodem<gf<8, 0x11D> , vector, logrealfast>::create);

template class direct_blockmodem<gf<9, 0x211> , vector, logrealfast> ;
template <>
const serializer
      direct_blockmodem<gf<9, 0x211> , vector, logrealfast>::shelper(
            "blockmodem", "blockmodem<gf<9,0x211>,logrealfast>",
            direct_blockmodem<gf<9, 0x211> , vector, logrealfast>::create);

template class direct_blockmodem<gf<10, 0x409> , vector, logrealfast> ;
template <>
const serializer
      direct_blockmodem<gf<10, 0x409> , vector, logrealfast>::shelper(
            "blockmodem", "blockmodem<gf<10,0x409>,logrealfast>",
            direct_blockmodem<gf<10, 0x409> , vector, logrealfast>::create);

template class direct_blockmodem<bool, vector, logrealfast> ;
template <>
const serializer direct_blockmodem<bool, vector, logrealfast>::shelper(
      "blockmodem", "blockmodem<bool,logrealfast>", direct_blockmodem<bool,
            vector, logrealfast>::create);

} // end namespace
