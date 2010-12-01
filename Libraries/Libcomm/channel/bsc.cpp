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

#include "bsc.h"
#include <sstream>

namespace libcomm {

const libbase::serializer bsc::shelper("channel", "bsc", bsc::create);

// Channel parameter handling

void bsc::set_parameter(const double Ps)
   {
   assert(Ps >=0 && Ps <= 0.5);
   bsc::Ps = Ps;
   }

// Channel function overrides

/*!
 * \copydoc channel::corrupt()
 * 
 * The channel model implemented is described by the following state diagram:
 * \dot
 * digraph bsidstates {
 * // Make figure left-to-right
 * rankdir = LR;
 * // state definitions
 * this [ shape=circle, color=gray, style=filled, label="t(i)" ];
 * next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
 * // path definitions
 * this -> next [ label="1-Ps" ];
 * this -> Substitute [ label="Ps" ];
 * Substitute -> next;
 * }
 * \enddot
 */
bool bsc::corrupt(const bool& s)
   {
   const double p = r.fval_closed();
   if (p < Ps)
      return !s;
   return s;
   }

// description output

std::string bsc::description() const
   {
   return "BSC channel";
   }

// object serialization - saving

std::ostream& bsc::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

std::istream& bsc::serialize(std::istream& sin)
   {
   return sin;
   }

} // end namespace
