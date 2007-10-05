#include "nrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

const libbase::vcs nrcc::version("Non-Recursive Convolutional Coder module (nrcc)", 1.50);

const libbase::serializer nrcc::shelper("fsm", "nrcc", nrcc::create);


// initialization

void nrcc::init(const libbase::matrix<bitfield>& generator)
   {
   // copy automatically what we can
   gen = generator;
   k = gen.xsize();
   n = gen.ysize();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and create shift registers
   reg = new bitfield[k];
   K = 0;
   for(int i=0; i<k; i++)
      {
      // assume with of register of input 'i' from its generator sequence for first output
      int m = gen(i, 0).size() - 1;
      reg[i].resize(m);
      K += m;
      // check that the gen. seq. for all outputs are the same length
      for(int j=1; j<n; j++)
         if(gen(i, j).size() != m+1)
            {
            std::cerr << "FATAL ERROR (nrcc): Generator sequence must have constant width for each input bit.\n";
            exit(1);
            }
      // update memory order
      if(m > nrcc::m)
         nrcc::m = m;
      }
   }

// constructors / destructors

nrcc::nrcc()
   {
   reg = NULL;
   }

nrcc::nrcc(const libbase::matrix<bitfield>& generator)
   {
   init(generator);
   }

nrcc::nrcc(const nrcc& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   K = x.K;
   m = x.m;
   gen = x.gen;
   // do the rest manually
   reg = new bitfield[k];
   for(int i=0; i<k; i++)
      reg[i] = x.reg[i];
   }

nrcc::~nrcc()
   {
   if(reg != NULL)
      delete[] reg;
   }

// finite state machine functions - resetting

void nrcc::reset(int state)
   {
   bitfield newstate;
   newstate.resize(K);
   newstate = state;
   for(int i=0; i<k; i++)
      {
      int size = reg[i].size();
      // check for case where no memory is associated with the input bit
      if(size > 0)
         {
         reg[i] = newstate.extract(size-1, 0);
         newstate >>= size;
         }
      }
   }

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
   assert("Function not implemented.");
   }

int nrcc::output(int& input)
   {
   assert("Function not implemented.");
   return 0;
   }

int nrcc::step(int& input)
   {
   bitfield ip, op;
   ip.resize(k);
   op.resize(0);
   // Handle tailing out
   if(input != fsm::tail)
      ip = input;
   else		// ip is the default of zero;
      input = 0;	// update the given input
   // Compute output
   for(int j=0; j<n; j++)
      {
      bitfield thisop;
      thisop.resize(1);
      for(int i=0; i<k; i++)
         thisop ^= (ip[i] + reg[i]) * gen(i, j);
      op = thisop + op;
      }
   // Compute next state
   for(int i=0; i<k; i++)
      reg[i] = ip[i] >> reg[i];
   return op;
   }

int nrcc::state() const
   {
   bitfield newstate;
   newstate.resize(0);
   for(int i=0; i<k; i++)
      {
      newstate = reg[i] + newstate;
      }
   return newstate;
   }

// description output

std::string nrcc::description() const
   {
   std::ostringstream sout;
   sout << "NRC code (K=" << m+1 << ", rate " << k << "/" << n << ", G=[";
   for(int i=0; i<k; i++)
      for(int j=0; j<n; j++)
         sout << gen(i, j) << (j==n-1 ? (i==k-1 ? "])" : "; ") : ", ");
   return sout.str();
   }

// object serialization - saving

std::ostream& nrcc::serialize(std::ostream& sout) const
   {
   sout << gen;
   return sout;
   }

// object serialization - loading

std::istream& nrcc::serialize(std::istream& sin)
   {
   sin >> gen;
   if(reg != NULL)
      delete[] reg;
   init(gen);
   return sin;
   }

}; // end namespace
