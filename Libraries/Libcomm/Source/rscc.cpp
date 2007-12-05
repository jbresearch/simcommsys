#include "rscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::vcs rscc::version("Recursive Systematic Convolutional Coder module (rscc)", 1.60);

const libbase::serializer rscc::shelper("fsm", "rscc", rscc::create);


// initialization

void rscc::init(const libbase::matrix<bitfield>& generator)
   {
   // copy automatically what we can
   gen = generator;
   k = gen.xsize();
   n = gen.ysize();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and create shift registers
   reg.init(k);
   nu = 0;
   for(int i=0; i<k; i++)
      {
      // assume with of register of input 'i' from its generator's denominator
      int m = gen(i,0).size() - 1;
      reg(i).resize(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for(int j=1; j<n; j++)
         if(gen(i,j).size() != m+1)
            {
            std::cerr << "FATAL ERROR (rscc): Generator sequence must have constant width for each input bit.\n";
            exit(1);
            }
      // update memory order
      if(m > rscc::m)
         rscc::m = m;
      }
   }

// constructors / destructors

rscc::rscc()
   {
   }

rscc::rscc(const libbase::matrix<bitfield>& generator)
   {
   init(generator);
   }

rscc::rscc(const rscc& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   nu = x.nu;
   m = x.m;
   gen = x.gen;
   reg = x.reg;
   }
   
rscc::~rscc()
   {
   }
   
// finite state machine functions - resetting

void rscc::reset(int state)
   {
   bitfield newstate;
   newstate.resize(nu);
   newstate = state;
   for(int i=0; i<k; i++)
      {
      int size = reg(i).size();
      // check for case where no memory is associated with the input bit
      if(size > 0)
         {
         reg(i) = newstate.extract(size-1, 0);
         newstate >>= size;
         }
      }
   }

void rscc::resetcircular(int zerostate, int n)
   {
   assert("Function not implemented.");
   }

void rscc::resetcircular()
   {
   assert("Function not implemented.");
   }

// finite state machine functions - state advance etc.

void rscc::advance(int& input)
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

int rscc::state() const
   {
   bitfield newstate;
   newstate.resize(0);
   for(int i=0; i<k; i++)
      newstate = reg(i) + newstate;
   return newstate;
   }

// description output

std::string rscc::description() const
   {
   std::ostringstream sout;
   sout << "RSC code (K=" << m+1 << ", rate " << k << "/" << n << ", G=[";
   for(int i=0; i<k; i++)
      for(int j=0; j<n; j++)
         sout << gen(i, j) << (j==n-1 ? (i==k-1 ? "])" : "; ") : ", ");
   return sout.str();
   }

// object serialization - saving

std::ostream& rscc::serialize(std::ostream& sout) const
   {
   sout << gen;
   return sout;
   }

// object serialization - loading

std::istream& rscc::serialize(std::istream& sin)
   {
   sin >> gen;
   init(gen);
   return sin;
   }

}; // end namespace
