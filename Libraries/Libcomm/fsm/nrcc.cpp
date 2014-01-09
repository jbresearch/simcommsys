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

#include "nrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;
using libbase::vector;

const libbase::serializer nrcc::shelper("fsm", "nrcc", nrcc::create);

// FSM state operations (getting and resetting)

void nrcc::resetcircular(const vector<int>& zerostate, int n)
   {
   failwith("Function not implemented.");
   }

// FSM helper operations

vector<int> nrcc::determineinput(const vector<int>& input) const
   {
   assert(input.size() == k);
   vector<int> ip = input;
   for (int i = 0; i < ip.size(); i++)
      if (ip(i) == fsm::tail)
         ip(i) = 0;
   return ip;
   }

bitfield nrcc::determinefeedin(const vector<int>& input) const
   {
   assert(input.size() == k);
   // check we have no 'tail' inputs
   for (int i = 0; i < k; i++)
      assert(input(i) != fsm::tail);
   // convert to required type
   return bitfield(vector<bool>(input));
   }

// Description

std::string nrcc::description() const
   {
   std::ostringstream sout;
   sout << "NRC code " << ccbfsm::description();
   return sout.str();
   }

// Serialization Support

std::ostream& nrcc::serialize(std::ostream& sout) const
   {
   return ccbfsm::serialize(sout);
   }

std::istream& nrcc::serialize(std::istream& sin)
   {
   return ccbfsm::serialize(sin);
   }

} // end namespace
