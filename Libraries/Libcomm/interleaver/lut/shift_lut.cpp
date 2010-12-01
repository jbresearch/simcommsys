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

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "shift_lut.h"
#include <sstream>

namespace libcomm {

// initialisation functions

template <class real>
void shift_lut<real>::init(const int amount, const int tau)
   {
   shift_lut<real>::amount = amount;

   this->lut.init(tau);
   for (int i = 0; i < tau; i++)
      this->lut(i) = (i + amount) % tau;
   }

// description output

template <class real>
std::string shift_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Shift by " << amount << " Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& shift_lut<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << std::endl;
   sout << amount << std::endl;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& shift_lut<real>::serialize(std::istream& sin)
   {
   int tau, amount;
   sin >> libbase::eatcomments >> tau >> amount;
   init(amount, tau);
   return sin;
   }

// Explicit instantiations

template class shift_lut<float> ;
template <>
const libbase::serializer shift_lut<float>::shelper("interleaver",
      "shift_lut<float>", shift_lut<float>::create);

template class shift_lut<double> ;
template <>
const libbase::serializer shift_lut<double>::shelper("interleaver",
      "shift_lut<double>", shift_lut<double>::create);

template class shift_lut<libbase::logrealfast> ;
template <>
const libbase::serializer shift_lut<libbase::logrealfast>::shelper(
      "interleaver", "shift_lut<logrealfast>",
      shift_lut<libbase::logrealfast>::create);

} // end namespace
