/*!
 * \file
 * 
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "direct_modem.h"
#include "gf.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

using libbase::gf;

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

template class direct_modem_implementation<gf<1, 0x3> > ;
template class direct_modem_implementation<gf<2, 0x7> > ;
template class direct_modem_implementation<gf<3, 0xB> > ;
template class direct_modem_implementation<gf<4, 0x13> > ;
template class direct_modem_implementation<gf<5, 0x25> > ;
template class direct_modem_implementation<gf<6, 0x43> > ;
template class direct_modem_implementation<gf<7, 0x89> > ;
template class direct_modem_implementation<gf<8, 0x11D> > ;
template class direct_modem_implementation<gf<9, 0x211> > ;
template class direct_modem_implementation<gf<10, 0x409> > ;

// *** Specific to direct_modem_implementation<bool> ***

// Description

std::string direct_modem_implementation<bool>::description() const
   {
   return "Binary Modulation";
   }

// Explicit Realizations

template class direct_modem<bool> ;
template class direct_modem<gf<1, 0x3> > ;
template class direct_modem<gf<2, 0x7> > ;
template class direct_modem<gf<3, 0xB> > ;
template class direct_modem<gf<4, 0x13> > ;
template class direct_modem<gf<5, 0x25> > ;
template class direct_modem<gf<6, 0x43> > ;
template class direct_modem<gf<7, 0x89> > ;
template class direct_modem<gf<8, 0x11D> > ;
template class direct_modem<gf<9, 0x211> > ;
template class direct_modem<gf<10, 0x409> > ;

} // end namespace
