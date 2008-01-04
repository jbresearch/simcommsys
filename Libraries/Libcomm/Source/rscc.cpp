/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "rscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::serializer rscc::shelper("fsm", "rscc", rscc::create);

// FSM state operations (getting and resetting)

void rscc::resetcircular(int zerostate, int n)
   {
   assertalways("Function not implemented.");
   }

void rscc::resetcircular()
   {
   assertalways("Function not implemented.");
   }

// FSM helper operations

bitfield rscc::determineinput(const int input) const
   {
   bitfield ip;
   if(input != fsm::tail)
      {
      ip.resize(k);
      ip = input;
      }
   else // Handle tailing out
      {
      ip.resize(0);
      for(int i=0; i<k; i++)
         ip = ((bitfield(0,1) + reg(i)) * gen(i,i)) + ip;
      }
   return ip;
   }

bitfield rscc::determinefeedin(const int input) const
   {
   assert(input != fsm::tail);
   bitfield sin(0,0), ip(input,k);
   for(int i=0; i<k; i++)
      sin = ((ip[i] + reg(i)) * gen(i,i)) + sin;
   return sin;
   }

// description output

std::string rscc::description() const
   {
   std::ostringstream sout;
   sout << "RSC code " << ccbfsm::description();
   return sout.str();
   }

}; // end namespace
