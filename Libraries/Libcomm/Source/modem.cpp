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

template class basic_modem< libbase::gf<1,0x3> >;
template class basic_modem< libbase::gf<2,0x7> >;
template class basic_modem< libbase::gf<3,0xB> >;
template class basic_modem< libbase::gf<4,0x13> >;
template class basic_modem<bool>;
template class basic_modem<libcomm::sigspace>;


// *** Templated GF(q) modem ***

// Description

template <class G>
std::string direct_modem<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << num_symbols() << ") Modulation";
   return sout.str();
   }

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

template class direct_modem< libbase::gf<1,0x3> >;
template <>
const libbase::serializer direct_modem< libbase::gf<1,0x3> >::shelper("modem", "modem<gf<1,0x3>>", direct_modem< libbase::gf<1,0x3> >::create);
template class direct_modem< libbase::gf<2,0x7> >;
template <>
const libbase::serializer direct_modem< libbase::gf<2,0x7> >::shelper("modem", "modem<gf<2,0x7>>", direct_modem< libbase::gf<2,0x7> >::create);
template class direct_modem< libbase::gf<3,0xB> >;
template <>
const libbase::serializer direct_modem< libbase::gf<3,0xB> >::shelper("modem", "modem<gf<3,0xB>>", direct_modem< libbase::gf<3,0xB> >::create);
template class direct_modem< libbase::gf<4,0x13> >;
template <>
const libbase::serializer direct_modem< libbase::gf<4,0x13> >::shelper("modem", "modem<gf<4,0x13>>", direct_modem< libbase::gf<4,0x13> >::create);


// *** Specific to direct_modem<bool> ***

const libbase::serializer direct_modem<bool>::shelper("modem", "modem<bool>", direct_modem<bool>::create);

// Description

std::string direct_modem<bool>::description() const
   {
   return "Binary Modulation";
   }

// Serialization Support

std::ostream& direct_modem<bool>::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& direct_modem<bool>::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace
