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

#include "flat.h"
#include <cstdlib>
#include <sstream>

namespace libcomm {

// initialization

template <class real>
void flat<real>::init(const int tau)
   {
   this->lut.init(tau);
   for (int i = 0; i < tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string flat<real>::description() const
   {
   std::ostringstream sout;
   sout << "Flat Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& flat<real>::serialize(std::ostream& sout) const
   {
   sout << "# Interleaver size" << std::endl;
   sout << this->lut.size() << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& flat<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> libbase::eatcomments >> tau;
   init(tau);
   return sin;
   }

// Explicit instantiations

template class flat<float> ;
template <>
const libbase::serializer flat<float>::shelper("interleaver", "flat<float>",
      flat<float>::create);

template class flat<double> ;
template <>
const libbase::serializer flat<double>::shelper("interleaver", "flat<double>",
      flat<double>::create);

template class flat<libbase::logrealfast> ;
template <>
const libbase::serializer flat<libbase::logrealfast>::shelper("interleaver",
      "flat<logrealfast>", flat<libbase::logrealfast>::create);

} // end namespace
