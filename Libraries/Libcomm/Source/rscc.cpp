#include "rscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::vcs rscc::version("Recursive Systematic Convolutional Coder module (rscc)", 1.70);

const libbase::serializer rscc::shelper("fsm", "rscc", rscc::create);


// constructors / destructors

rscc::rscc()
   {
   }

rscc::rscc(const libbase::matrix<bitfield>& generator) : ccbfsm(generator)
   {
   }

rscc::rscc(const rscc& x) : ccbfsm(x)
   {
   }
   
rscc::~rscc()
   {
   }
   
// finite state machine functions - resetting

void rscc::resetcircular(int zerostate, int n)
   {
   assert("Function not implemented.");
   }

void rscc::resetcircular()
   {
   assert("Function not implemented.");
   }

// finite state machine functions - state advance etc.

bitfield rscc::determinefeedin(int &input)
   {
   bitfield ip, op;
   // For RSC, ip holds the value after the adder, not the actual i/p
   // and the first (low-order) k bits of the o/p are merely copies of the i/p
   if(input != fsm::tail)
      {
      ip.resize(0);
      op.resize(k);
      op = input;
      for(int i=0; i<k; i++)
         ip = ((op[i] + reg(i)) * gen(i,i)) + ip;
      }
   else // Handle tailing out
      {
      ip.resize(k);
      op.resize(0);
      for(int i=0; i<k; i++)
         op = ((ip[i] + reg(i)) * gen(i,i)) + op;
      input = op;               // update given input as necessary
      }
   return ip;
   }

void rscc::advance(int& input)
   {
   bitfield ip = determinefeedin(input);
   // Compute next state
   for(int i=0; i<k; i++)
      reg(i) = ip[i] >> reg(i);
   }

int rscc::output(const int& input) const
   {
   bitfield ip, op;
   // For RSC, ip holds the value after the adder, not the actual i/p
   // and the first (low-order) k bits of the o/p are merely copies of the i/p
   if(input != fsm::tail)
      {
      ip.resize(0);
      op.resize(k);
      op = input;
      for(int i=0; i<k; i++)
         ip = ((op[i] + reg(i)) * gen(i,i)) + ip;
      }
   else // Handle tailing out
      {
      ip.resize(k);
      op.resize(0);
      for(int i=0; i<k; i++)
         op = ((ip[i] + reg(i)) * gen(i,i)) + op;
      }
   // Compute output
   for(int j=k; j<n; j++)
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

std::string rscc::description() const
   {
   std::ostringstream sout;
   sout << "RSC code " << ccbfsm::description();
   return sout.str();
   }

// object serialization - saving

std::ostream& rscc::serialize(std::ostream& sout) const
   {
   return ccbfsm::serialize(sout);
   }

// object serialization - loading

std::istream& rscc::serialize(std::istream& sin)
   {
   return ccbfsm::serialize(sin);
   }

}; // end namespace
