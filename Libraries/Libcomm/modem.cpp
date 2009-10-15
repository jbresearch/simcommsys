/*!
 * \file
 * 
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "modem.h"
#include "gf.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// *** Common Modulator Interface ***

using libbase::gf;

// Explicit Realizations

template class basic_modem<gf<1, 0x3> > ;
template class basic_modem<gf<2, 0x7> > ;
template class basic_modem<gf<3, 0xB> > ;
template class basic_modem<gf<4, 0x13> > ;
template class basic_modem<gf<5, 0x25> > ;
template class basic_modem<gf<6, 0x43> > ;
template class basic_modem<gf<7, 0x89> > ;
template class basic_modem<gf<8, 0x11D> > ;
template class basic_modem<gf<9, 0x211> > ;
template class basic_modem<gf<10, 0x409> > ;
template class basic_modem<bool> ;
template class basic_modem<sigspace> ;

} // end namespace
