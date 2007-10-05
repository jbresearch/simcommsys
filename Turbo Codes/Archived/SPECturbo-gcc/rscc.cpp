#include "rscc.h"

#include <iostream.h>

const vcs rscc_version("Recursive Systematic Convolutional Coder module (rscc)", 1.00);

rscc::rscc(const int k, const int n, const matrix<bitfield> generator)
   {
   // copy automatically what we can
   rscc::k = k;
   rscc::n = n;
   gen = generator;
   // set default value to the rest
   rscc::m = 0;
   // check that the generator matrix is valid (correct sizes) and create shift registers
   reg = new bitfield[k];
   K = 0;
   for(int i=0; i<k; i++)
      {
      // assume with of register of input 'i' from its generator's denominator
      int m = gen(i, 0).size() - 1;
      reg[i].resize(m);
      K += m;
      // check that the gen. seq. for all outputs are the same length
      for(int j=1; j<n; j++)
         if(gen(i, j).size() != m+1)
            {
            cerr << "FATAL ERROR (rscc): Generator sequence must have constant width for each input bit.\n";
            exit(1);
            }
      // update memory order
      if(m > rscc::m)
         rscc::m = m;
      }
   }

rscc::rscc(const rscc& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   K = x.K;
   gen = x.gen;
   // do the rest manually
   reg = new bitfield[k];
   for(int i=0; i<k; i++)
      reg[i] = x.reg[i];
   }
   
rscc::~rscc()
   {
   delete[] reg;
   }
   
fsm *rscc::clone()
   {
   return new rscc(*this);
   }

void rscc::reset(int state)
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

int rscc::step(int& input)
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
         ip = (((op[i], reg[i]) * gen(i, i)), ip);
      }
   else // Handle tailing out
      {
      ip.resize(k);
      op.resize(0);
      for(int i=0; i<k; i++)
         op = (((ip[i], reg[i]) * gen(i, i)), op);
      input = op;		// update given input as necessary
      }
   // Compute output
   for(int j=k; j<n; j++)
      {
      bitfield thisop;
      thisop.resize(1);
      for(int i=0; i<k; i++)
         thisop ^= (ip[i], reg[i]) * gen(i, j);
      op = (thisop, op);
      }
   // Compute next state
   for(int i=0; i<k; i++)
      reg[i] = ip[i] >> reg[i];
   return op;
   }

int rscc::state() const
   {
   bitfield newstate;
   newstate.resize(0);
   for(int i=0; i<k; i++)
      newstate = reg[i], newstate;
   return newstate;
   }

void rscc::print(ostream& s) const
   {
   s << "RSC code (K=" << m+1 << ", rate " << k << "/" << n << ", G=[";
   for(int i=0; i<k; i++)
      for(int j=0; j<n; j++)
         s << gen(i, j) << (j==n-1 ? (i==k-1 ? "])" : "; ") : ", ");
   }

