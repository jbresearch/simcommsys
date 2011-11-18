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

#include "commsys_threshold.h"

#include <sstream>

namespace libcomm {

// Experiment parameter handling

template <class S, class R>
void commsys_threshold<S, R>::set_parameter(const double x)
   {
   parametric& m = dynamic_cast<parametric&> (*this->sys->getmodem());
   m.set_parameter(x);
   }

template <class S, class R>
double commsys_threshold<S, R>::get_parameter() const
   {
   const parametric& m =
         dynamic_cast<const parametric&> (*this->sys->getmodem());
   return m.get_parameter();
   }

// Description & Serialization

template <class S, class R>
std::string commsys_threshold<S, R>::description() const
   {
   std::ostringstream sout;
   sout << "Modem-threshold-varying ";
   sout << Base::description();
   return sout.str();
   }

template <class S, class R>
std::ostream& commsys_threshold<S, R>::serialize(std::ostream& sout) const
   {
   sout << this->sys->getchan()->get_parameter() << std::endl;
   Base::serialize(sout);
   return sout;
   }

template <class S, class R>
std::istream& commsys_threshold<S, R>::serialize(std::istream& sin)
   {
   double x;
   sin >> libbase::eatcomments >> x;
   Base::serialize(sin);
   this->sys->getchan()->set_parameter(x);
   return sin;
   }

// Explicit Realizations

template class commsys_threshold<bool> ;
template <>
const libbase::serializer commsys_threshold<bool>::shelper("experiment",
      "commsys_threshold<bool>", commsys_threshold<bool>::create);

} // end namespace
