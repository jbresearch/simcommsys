/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "nrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::serializer nrcc::shelper("fsm", "nrcc", nrcc::create);

// FSM state operations (getting and resetting)

void nrcc::resetcircular(int zerostate, int n)
   {
   failwith("Function not implemented.");
   }

// FSM helper operations

bitfield nrcc::determineinput(const int input) const
   {
   bitfield ip(0, k);
   if (input != fsm::tail)
      ip = input;
   return ip;
   }

bitfield nrcc::determinefeedin(const int input) const
   {
   assert(input != fsm::tail);
   return bitfield(input, k);
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
