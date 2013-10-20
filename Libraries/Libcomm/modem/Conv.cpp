/*!
 * \file
 * $Id: qam.cpp 9469 2013-07-24 16:38:03Z jabriffa $
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
 */

#include "conv.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

const libbase::serializer conv::shelper("blockmodem", "conv", conv::create);

// Internal operations

/*! \brief Initialization
 *
 * For square constellations, allocate symbols using Gray code sequences for
 * rows and columns (i.e adjacent symbols in the quadrature and in-phase directions
 * respectively); each symbol then represents the concatenation of these two Gray
 * code values.
 *
 * The constellation is centered about the origin, and uses the conventional
 * scaling where symbols take amplitudes ± 1, 3, 5, ...
 *
 * \sa Sklar, 2nd edition, p.565.
 *
 * \sa Matlab bin2gray function from the communications toolbox
 */
void conv::init(const int m)
   {
   }

// description output

void conv::domodulate(const int N, const array1i_t& encoded, array1s_t& tx)
   {

   }

std::string conv::description() const
   {
   std::ostringstream sout;
   sout << "Gray " << lut.size() << "QAM blockmodem";
   return sout.str();
   }

// object serialization - saving

std::ostream& conv::serialize(std::ostream& sout) const
   {
   sout << lut.size() << std::endl;
   return sout;
   }

// object serialization - loading

std::istream& conv::serialize(std::istream& sin)
   {
   int m;
   sin >> libbase::eatcomments >> m >> libbase::verify;
   init(m);
   return sin;
   }

} // end namespace
