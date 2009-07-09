/*!
 \file

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$
 */

#include "modem.h"
#include "gf.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// *** Common Modulator Interface ***

// Explicit Realizations

template class basic_modem<libbase::gf<1, 0x3> > ;
template class basic_modem<libbase::gf<2, 0x7> > ;
template class basic_modem<libbase::gf<3, 0xB> > ;
template class basic_modem<libbase::gf<4, 0x13> > ;
template class basic_modem<bool> ;
template class basic_modem<libcomm::sigspace> ;
// *** Templated GF(q) modem ***

// Description

template <class G>
std::string direct_modem_implementation<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << num_symbols() << ") Modulation";
   return sout.str();
   }

// Explicit Realizations

template class direct_modem_implementation<libbase::gf<1, 0x3> > ;
template class direct_modem_implementation<libbase::gf<2, 0x7> > ;
template class direct_modem_implementation<libbase::gf<3, 0xB> > ;
template class direct_modem_implementation<libbase::gf<4, 0x13> > ;
// *** Specific to direct_modem_implementation<bool> ***

// Description

std::string direct_modem_implementation<bool>::description() const
   {
   return "Binary Modulation";
   }

// *** Templated Direct Modem ***

// Serialization Support

template <class G>
std::ostream& direct_modem<G>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class G>
std::istream& direct_modem<G>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit Realizations

template class direct_modem<bool> ;
template <>
const libbase::serializer direct_modem<bool>::shelper("modem", "modem<bool>",
      direct_modem<bool>::create);

template class direct_modem<libbase::gf<1, 0x3> > ;
template <>
const libbase::serializer direct_modem<libbase::gf<1, 0x3> >::shelper("modem",
      "modem<gf<1,0x3>>", direct_modem<libbase::gf<1, 0x3> >::create);
template class direct_modem<libbase::gf<2, 0x7> > ;
template <>
const libbase::serializer direct_modem<libbase::gf<2, 0x7> >::shelper("modem",
      "modem<gf<2,0x7>>", direct_modem<libbase::gf<2, 0x7> >::create);
template class direct_modem<libbase::gf<3, 0xB> > ;
template <>
const libbase::serializer direct_modem<libbase::gf<3, 0xB> >::shelper("modem",
      "modem<gf<3,0xB>>", direct_modem<libbase::gf<3, 0xB> >::create);
template class direct_modem<libbase::gf<4, 0x13> > ;
template <>
const libbase::serializer direct_modem<libbase::gf<4, 0x13> >::shelper("modem",
      "modem<gf<4,0x13>>", direct_modem<libbase::gf<4, 0x13> >::create);

} // end namespace
