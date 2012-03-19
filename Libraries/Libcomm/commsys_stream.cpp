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

#include "commsys_stream.h"

#include <sstream>

namespace libcomm {

// Communication System Interface

template <class S, template <class > class C>
void commsys_stream<S, C>::receive_path(const C<S>& received,
      const C<double>& sof_prior, const C<double>& eof_prior,
      const libbase::size_type<C> offset)
   {
   // Get access to the commsys modem in stream-oriented mode
   stream_modulator<S, C>& m = getmodem_stream();
   // Demodulate
   C<array1d_t> ptable_mapped;
   m.reset_timers();
   m.demodulate(*this->chan, received, sof_prior, eof_prior, ptable_mapped,
         ptable_mapped, sof_post, eof_post, offset);
   this->add_timers(m);
   // After-demodulation receive path
   Base::softreceive_path(ptable_mapped);
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_stream<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Stream-oriented ";
   sout << Base::description();
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_stream<S, C>::serialize(std::ostream& sout) const
   {
   Base::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_stream<S, C>::serialize(std::istream& sin)
   {
   Base::serialize(sin);
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations

using libbase::serializer;
using libbase::gf;

template class commsys_stream<sigspace> ;
template <>
const serializer commsys_stream<sigspace>::shelper("commsys",
      "commsys_stream<sigspace>", commsys_stream<sigspace>::create);

template class commsys_stream<bool> ;
template <>
const serializer commsys_stream<bool>::shelper("commsys",
      "commsys_stream<bool>", commsys_stream<bool>::create);

template class commsys_stream<gf<1, 0x3> > ;
template <>
const serializer commsys_stream<gf<1, 0x3> >::shelper(
      "commsys", "commsys_stream<gf<1,0x3>>", commsys_stream<
            gf<1, 0x3> >::create);

template class commsys_stream<gf<2, 0x7> > ;
template <>
const serializer commsys_stream<gf<2, 0x7> >::shelper(
      "commsys", "commsys_stream<gf<2,0x7>>", commsys_stream<
            gf<2, 0x7> >::create);

template class commsys_stream<gf<3, 0xB> > ;
template <>
const serializer commsys_stream<gf<3, 0xB> >::shelper(
      "commsys", "commsys_stream<gf<3,0xB>>", commsys_stream<
            gf<3, 0xB> >::create);

template class commsys_stream<gf<4, 0x13> > ;
template <>
const serializer commsys_stream<gf<4, 0x13> >::shelper(
      "commsys", "commsys_stream<gf<4,0x13>>", commsys_stream<gf<4,
            0x13> >::create);

template class commsys_stream<gf<5, 0x25> > ;
template <>
const serializer commsys_stream<gf<5, 0x25> >::shelper(
      "commsys", "commsys_stream<gf<5,0x25>>", commsys_stream<gf<5,
            0x25> >::create);

template class commsys_stream<gf<6, 0x43> > ;
template <>
const serializer commsys_stream<gf<6, 0x43> >::shelper(
      "commsys", "commsys_stream<gf<6,0x43>>", commsys_stream<gf<6,
            0x43> >::create);

template class commsys_stream<gf<7, 0x89> > ;
template <>
const serializer commsys_stream<gf<7, 0x89> >::shelper(
      "commsys", "commsys_stream<gf<7,0x89>>", commsys_stream<gf<7,
            0x89> >::create);

template class commsys_stream<gf<8, 0x11D> > ;
template <>
const serializer commsys_stream<gf<8, 0x11D> >::shelper(
      "commsys", "commsys_stream<gf<8,0x11D>>", commsys_stream<gf<8,
            0x11D> >::create);

template class commsys_stream<gf<9, 0x211> > ;
template <>
const serializer commsys_stream<gf<9, 0x211> >::shelper(
      "commsys", "commsys_stream<gf<9,0x211>>", commsys_stream<gf<9,
            0x211> >::create);

template class commsys_stream<gf<10, 0x409> > ;
template <>
const serializer commsys_stream<gf<10, 0x409> >::shelper(
      "commsys", "commsys_stream<gf<10,0x409>>", commsys_stream<gf<10,
            0x409> >::create);

} // end namespace
