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

#include "commsys_timer.h"
#include "gf.h"

namespace libcomm {

// Serialization Support

template <class S>
std::ostream& commsys_timer<S>::serialize(std::ostream& sout) const
   {
   simulator.serialize(sout);
   return sout;
   }

template <class S>
std::istream& commsys_timer<S>::serialize(std::istream& sin)
   {
   simulator.serialize(sin);
   return sin;
   }

// Explicit Realizations

using libbase::serializer;
using libbase::gf;

template class commsys_timer<sigspace> ;
template <>
const serializer commsys_timer<sigspace>::shelper("experiment",
      "commsys_timer<sigspace>", commsys_timer<sigspace>::create);

template class commsys_timer<bool> ;
template <>
const serializer commsys_timer<bool>::shelper("experiment",
      "commsys_timer<bool>", commsys_timer<bool>::create);

template class commsys_timer<gf<1, 0x3> > ;
template <>
const serializer commsys_timer<gf<1, 0x3> >::shelper("experiment",
      "commsys_timer<gf<1,0x3>>", commsys_timer<gf<1, 0x3> >::create);

template class commsys_timer<gf<2, 0x7> > ;
template <>
const serializer commsys_timer<gf<2, 0x7> >::shelper("experiment",
      "commsys_timer<gf<2,0x7>>", commsys_timer<gf<2, 0x7> >::create);

template class commsys_timer<gf<3, 0xB> > ;
template <>
const serializer commsys_timer<gf<3, 0xB> >::shelper("experiment",
      "commsys_timer<gf<3,0xB>>", commsys_timer<gf<3, 0xB> >::create);

template class commsys_timer<gf<4, 0x13> > ;
template <>
const serializer commsys_timer<gf<4, 0x13> >::shelper("experiment",
      "commsys_timer<gf<4,0x13>>", commsys_timer<gf<4, 0x13> >::create);

template class commsys_timer<gf<5, 0x25> > ;
template <>
const serializer commsys_timer<gf<5, 0x25> >::shelper("experiment",
      "commsys_timer<gf<5,0x25>>", commsys_timer<gf<5, 0x25> >::create);

template class commsys_timer<gf<6, 0x43> > ;
template <>
const serializer commsys_timer<gf<6, 0x43> >::shelper("experiment",
      "commsys_timer<gf<6,0x43>>", commsys_timer<gf<6, 0x43> >::create);

template class commsys_timer<gf<7, 0x89> > ;
template <>
const serializer commsys_timer<gf<7, 0x89> >::shelper("experiment",
      "commsys_timer<gf<7,0x89>>", commsys_timer<gf<7, 0x89> >::create);

template class commsys_timer<gf<8, 0x11D> > ;
template <>
const serializer commsys_timer<gf<8, 0x11D> >::shelper("experiment",
      "commsys_timer<gf<8,0x11D>>", commsys_timer<gf<8, 0x11D> >::create);

template class commsys_timer<gf<9, 0x211> > ;
template <>
const serializer commsys_timer<gf<9, 0x211> >::shelper("experiment",
      "commsys_timer<gf<9,0x211>>", commsys_timer<gf<9, 0x211> >::create);

template class commsys_timer<gf<10, 0x409> > ;
template <>
const serializer commsys_timer<gf<10, 0x409> >::shelper("experiment",
      "commsys_timer<gf<10,0x409>>", commsys_timer<gf<10, 0x409> >::create);

} // end namespace
