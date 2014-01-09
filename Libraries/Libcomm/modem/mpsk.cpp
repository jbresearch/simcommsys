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
 */

#include "mpsk.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

const libbase::serializer mpsk::shelper("blockmodem", "mpsk", mpsk::create);

// initialization

void mpsk::init(const int m)
   {
   lut.init(m);
   // allocate symbols using a Gray code sequence
   using libbase::gray;
   for (int i = 0; i < m; i++)
      {
      const double r = 1;
      const double theta = i * (2 * libbase::PI / m);
      lut(gray(i)) = sigspace(r * cos(theta), r * sin(theta));
      }
   }

// description output

std::string mpsk::description() const
   {
   std::ostringstream sout;
   switch (lut.size())
      {
      case 2:
         sout << "BPSK blockmodem";
         break;
      case 4:
         sout << "Gray QPSK blockmodem";
         break;
      default:
         sout << "Gray " << lut.size() << "PSK blockmodem";
         break;
      }
   return sout.str();
   }

// object serialization - saving

std::ostream& mpsk::serialize(std::ostream& sout) const
   {
   sout << "# Alphabet size in symbols" << std::endl;
   sout << lut.size() << std::endl;
   return sout;
   }

// object serialization - loading

std::istream& mpsk::serialize(std::istream& sin)
   {
   int m;
   sin >> libbase::eatcomments >> m >> libbase::verify;
   init(m);
   return sin;
   }

} // end namespace
