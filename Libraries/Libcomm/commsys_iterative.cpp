/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
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
   sout << iter;
   commsys<S, C>::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_iterative<S, C>::serialize(std::istream& sin)
   {
   sin >> iter;
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

} // end namespace
