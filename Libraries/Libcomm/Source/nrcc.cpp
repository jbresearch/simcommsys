#include "nrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::vcs nrcc::version("Non-Recursive Convolutional Coder module (nrcc)", 1.70);

const libbase::serializer nrcc::shelper("fsm", "nrcc", nrcc::create);


// constructors / destructors

nrcc::nrcc()
   {
   }

nrcc::nrcc(const libbase::matrix<bitfield>& generator) : ccbfsm(generator)
   {
   }

nrcc::nrcc(const nrcc& x) : ccbfsm(x)
   {
   }

nrcc::~nrcc()
   {
   }

// finite state machine functions - resetting

void nrcc::resetcircular(int zerostate, int n)
   {
   assert("Function not implemented.");
   }

void nrcc::resetcircular()
   {
   assert("Function not implemented.");
   }

// finite state machine functions - state advance etc.

void nrcc::advance(int& input)
   {
   bitfield ip;
   ip.resize(k);
   // Handle tailing out
   if(input != fsm::tail)
      ip = input;
   else         // ip is the default of zero;
      input = 0;        // update the given input
   // Compute next state
   for(int i=0; i<k; i++)
      reg(i) = ip[i] >> reg(i);
   }

int nrcc::output(const int& input) const
   {
   bitfield ip, op;
   ip.resize(k);
   op.resize(0);
   // Handle tailing out
   if(input != fsm::tail)
      ip = input;
   // Compute output
   for(int j=0; j<n; j++)
      {
      bitfield thisop;
      thisop.resize(1);
      for(int i=0; i<k; i++)
         thisop ^= (ip[i] + reg(i)) * gen(i,j);
      op = thisop + op;
      }
   return op;
   }

// description output

std::string nrcc::description() const
   {
   std::ostringstream sout;
   sout << "NRC code " << ccbfsm::description();
   return sout.str();
   }

// object serialization - saving

std::ostream& nrcc::serialize(std::ostream& sout) const
   {
   return ccbfsm::serialize(sout);
   }

// object serialization - loading

std::istream& nrcc::serialize(std::istream& sin)
   {
   return ccbfsm::serialize(sin);
   }

}; // end namespace
