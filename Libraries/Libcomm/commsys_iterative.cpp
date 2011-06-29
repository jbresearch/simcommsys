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

#include "commsys_iterative.h"

#include "modem/informed_modulator.h"
#include "gf.h"
#include <sstream>

namespace libcomm {

// Communication System Interface

template <class S, template <class > class C>
void commsys_iterative<S, C>::receive_path(const C<S>& received)
   {
   // Demodulate
   C<array1d_t> ptable_mapped;
   informed_modulator<S>& m = dynamic_cast<informed_modulator<S>&> (*this->mdm);
   for (int i = 0; i < iter; i++)
      {
      libbase::trace
            << "DEBUG (commsys_iterative): Starting demodulation iteration "
            << i << std::endl;
      m.demodulate(*this->chan, received, ptable_mapped, ptable_mapped);
      m.mark_as_clean();
      }
   m.mark_as_dirty();
   // After-demodulation receive path
   softreceive_path(ptable_mapped);
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_iterative<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative ";
   sout << commsys<S, C>::description() << ", ";
   sout << iter << " iterations";
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_iterative<S, C>::serialize(std::ostream& sout) const
   {
   sout << iter << std::endl;
   commsys<S, C>::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_iterative<S, C>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter;
   commsys<S, C>::serialize(sin);
   return sin;
   }

// Explicit Realizations

template class commsys_iterative<sigspace> ;
template <>
const libbase::serializer commsys_iterative<sigspace>::shelper("commsys",
      "commsys_iterative<sigspace>", commsys_iterative<sigspace>::create);

template class commsys_iterative<bool> ;
template <>
const libbase::serializer commsys_iterative<bool>::shelper("commsys",
      "commsys_iterative<bool>", commsys_iterative<bool>::create);

template class commsys_iterative<libbase::gf<1, 0x3> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<1, 0x3> >::shelper(
      "commsys", "commsys_iterative<gf<1,0x3>>", commsys_iterative<libbase::gf<
            1, 0x3> >::create);

template class commsys_iterative<libbase::gf<2, 0x7> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<2, 0x7> >::shelper(
      "commsys", "commsys_iterative<gf<2,0x7>>", commsys_iterative<libbase::gf<
            2, 0x7> >::create);

template class commsys_iterative<libbase::gf<3, 0xB> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<3, 0xB> >::shelper(
      "commsys", "commsys_iterative<gf<3,0xB>>", commsys_iterative<libbase::gf<
            3, 0xB> >::create);

template class commsys_iterative<libbase::gf<4, 0x13> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<4, 0x13> >::shelper(
      "commsys", "commsys_iterative<gf<4,0x13>>", commsys_iterative<
            libbase::gf<4, 0x13> >::create);

template class commsys_iterative<libbase::gf<5, 0x25> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<5, 0x25> >::shelper(
      "commsys", "commsys_iterative<gf<5,0x25>>", commsys_iterative<
            libbase::gf<5, 0x25> >::create);

template class commsys_iterative<libbase::gf<6, 0x43> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<6, 0x43> >::shelper(
      "commsys", "commsys_iterative<gf<6,0x43>>", commsys_iterative<
            libbase::gf<6, 0x43> >::create);

template class commsys_iterative<libbase::gf<7, 0x89> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<7, 0x89> >::shelper(
      "commsys", "commsys_iterative<gf<7,0x89>>", commsys_iterative<
            libbase::gf<7, 0x89> >::create);

template class commsys_iterative<libbase::gf<8, 0x11D> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<8, 0x11D> >::shelper(
      "commsys", "commsys_iterative<gf<8,0x11D>>", commsys_iterative<
            libbase::gf<8, 0x11D> >::create);

template class commsys_iterative<libbase::gf<9, 0x211> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<9, 0x211> >::shelper(
      "commsys", "commsys_iterative<gf<9,0x211>>", commsys_iterative<
            libbase::gf<9, 0x211> >::create);

template class commsys_iterative<libbase::gf<10, 0x409> > ;
template <>
const libbase::serializer commsys_iterative<libbase::gf<10, 0x409> >::shelper(
      "commsys", "commsys_iterative<gf<10,0x409>>", commsys_iterative<
            libbase::gf<10, 0x409> >::create);

} // end namespace
