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

void nrcc::resetcircular(libbase::vector<int> zerostate, int n)
   {
   failwith("Function not implemented.");
   }

// FSM helper operations

bitfield nrcc::determineinput(libbase::vector<int> input) const
   {
   assert(input.size() == k);
   // replace 'tail' inputs with zeros
   for (int i = 0; i < k; i++)
      if (input(i) == fsm::tail)
         input(i) = 0;
   // convert to required type
   return libbase::vector<bool>(input);
   }

bitfield nrcc::determinefeedin(libbase::vector<int> input) const
   {
   assert(input.size() == k);
   // check we have no 'tail' inputs
   for (int i = 0; i < k; i++)
      assert(input(i) != fsm::tail);
   // convert to required type
   return libbase::vector<bool>(input);
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
