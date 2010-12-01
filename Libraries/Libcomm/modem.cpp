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

#include "modem.h"
#include "gf.h"
#include <cstdlib>
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
