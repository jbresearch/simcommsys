/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "commsys_fulliter.h"

#include "modem/informed_modulator.h"
#include "gf.h"
#include <sstream>

namespace libcomm {

// Communication System Interface

template <class S, template <class > class C>
void commsys_fulliter<S, C>::receive_path(const C<S>& received)
   {
   // Demodulate
   C<array1d_t> ptable_mapped;
   informed_modulator<S>& m = dynamic_cast<informed_modulator<S>&> (*this->mdm);
   for (int i = 0; i < iter; i++)
      {
      libbase::trace
            << "DEBUG (commsys_fulliter): Starting demodulation iteration "
            << i << "\n";
      m.demodulate(*this->chan, received, ptable_mapped, ptable_mapped);
      m.mark_as_clean();
      }
   m.mark_as_dirty();
   // Inverse Map
   C<array1d_t> ptable_encoded;
   this->map->inverse(ptable_mapped, ptable_encoded);
   // Translate
   this->cdc->init_decoder(ptable_encoded);
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_fulliter<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative ";
   sout << commsys<S, C>::description() << ", ";
   sout << iter << " iterations";
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_fulliter<S, C>::serialize(std::ostream& sout) const
   {
   sout << iter;
   commsys<S, C>::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_fulliter<S, C>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter;
   commsys<S, C>::serialize(sin);
   return sin;
   }

// Explicit Realizations

template class commsys_fulliter<sigspace> ;
template <>
const libbase::serializer commsys_fulliter<sigspace>::shelper("commsys",
      "commsys_fulliter<sigspace>", commsys_fulliter<sigspace>::create);

template class commsys_fulliter<bool> ;
template <>
const libbase::serializer commsys_fulliter<bool>::shelper("commsys",
      "commsys_fulliter<bool>", commsys_fulliter<bool>::create);

template class commsys_fulliter<libbase::gf<1, 0x3> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<1, 0x3> >::shelper(
      "commsys", "commsys_fulliter<gf<1,0x3>>", commsys_fulliter<libbase::gf<
            1, 0x3> >::create);

template class commsys_fulliter<libbase::gf<2, 0x7> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<2, 0x7> >::shelper(
      "commsys", "commsys_fulliter<gf<2,0x7>>", commsys_fulliter<libbase::gf<
            2, 0x7> >::create);

template class commsys_fulliter<libbase::gf<3, 0xB> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<3, 0xB> >::shelper(
      "commsys", "commsys_fulliter<gf<3,0xB>>", commsys_fulliter<libbase::gf<
            3, 0xB> >::create);

template class commsys_fulliter<libbase::gf<4, 0x13> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<4, 0x13> >::shelper(
      "commsys", "commsys_fulliter<gf<4,0x13>>", commsys_fulliter<
            libbase::gf<4, 0x13> >::create);

template class commsys_fulliter<libbase::gf<5, 0x25> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<5, 0x25> >::shelper(
      "commsys", "commsys_fulliter<gf<5,0x25>>", commsys_fulliter<
            libbase::gf<5, 0x25> >::create);

template class commsys_fulliter<libbase::gf<6, 0x43> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<6, 0x43> >::shelper(
      "commsys", "commsys_fulliter<gf<6,0x43>>", commsys_fulliter<
            libbase::gf<6, 0x43> >::create);

template class commsys_fulliter<libbase::gf<7, 0x89> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<7, 0x89> >::shelper(
      "commsys", "commsys_fulliter<gf<7,0x89>>", commsys_fulliter<
            libbase::gf<7, 0x89> >::create);

template class commsys_fulliter<libbase::gf<8, 0x11D> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<8, 0x11D> >::shelper(
      "commsys", "commsys_fulliter<gf<8,0x11D>>", commsys_fulliter<
            libbase::gf<8, 0x11D> >::create);

template class commsys_fulliter<libbase::gf<9, 0x211> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<9, 0x211> >::shelper(
      "commsys", "commsys_fulliter<gf<9,0x211>>", commsys_fulliter<
            libbase::gf<9, 0x211> >::create);

template class commsys_fulliter<libbase::gf<10, 0x409> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<10, 0x409> >::shelper(
      "commsys", "commsys_fulliter<gf<10,0x409>>", commsys_fulliter<
            libbase::gf<10, 0x409> >::create);

} // end namespace
