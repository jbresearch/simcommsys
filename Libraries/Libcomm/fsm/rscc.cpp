/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "rscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;
using libbase::vector;

const libbase::serializer rscc::shelper("fsm", "rscc", rscc::create);

// FSM state operations (getting and resetting)

void rscc::resetcircular(const vector<int>& zerostate, int n)
   {
   failwith("Function not implemented.");
   }

// FSM helper operations

vector<int> rscc::determineinput(const vector<int>& input) const
   {
   assert(input.size() == k);
   vector<int> ip = input;
   for (int i = 0; i < ip.size(); i++)
      if (ip(i) == fsm::tail)
         ip(i) = (reg(i) + bitfield(0, 1)) * gen(i, i);
   return ip;
   }

bitfield rscc::determinefeedin(const vector<int>& input) const
   {
   assert(input.size() == k);
   // check we have no 'tail' inputs
   for (int i = 0; i < k; i++)
      assert(input(i) != fsm::tail);
   // compute input junction
   bitfield sin(0, 0), ip = bitfield(vector<bool>(input));
   for (int i = 0; i < k; i++)
      sin = ((reg(i) + ip(i)) * gen(i, i)) + sin;
   return sin;
   }

// Description

std::string rscc::description() const
   {
   std::ostringstream sout;
   sout << "RSC code " << ccbfsm::description();
   return sout.str();
   }

// Serialization Support

std::ostream& rscc::serialize(std::ostream& sout) const
   {
   return ccbfsm::serialize(sout);
   }

std::istream& rscc::serialize(std::istream& sin)
   {
   return ccbfsm::serialize(sin);
   }

} // end namespace
