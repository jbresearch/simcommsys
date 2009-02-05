/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "ccbfsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

// initialization

void ccbfsm::init(const libbase::matrix<bitfield>& generator)
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
      // assume with of register of input 'i' from its generator sequence for first output
      int m = gen(i,0).size() - 1;
      reg(i).resize(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for(int j=1; j<n; j++)
         if(gen(i,j).size() != m+1)
            {
            std::cerr << "FATAL ERROR (ccbfsm): Generator sequence must have constant width for each input bit.\n";
            exit(1);
            }
      // update memory order
      if(m > ccbfsm::m)
         ccbfsm::m = m;
      }
   }

// constructors / destructors

ccbfsm::ccbfsm(const libbase::matrix<bitfield>& generator)
   {
   init(generator);
   }

ccbfsm::ccbfsm(const ccbfsm& x)
   {
   // copy automatically what we can
   k = x.k;
   n = x.n;
   nu = x.nu;
   m = x.m;
   gen = x.gen;
   reg = x.reg;
   }

// FSM state operations (getting and resetting)

int ccbfsm::state() const
   {
   bitfield newstate;
   newstate.resize(0);
   for(int i=0; i<k; i++)
      {
      newstate = reg(i) + newstate;
      }
   return newstate;
   }

void ccbfsm::reset(int state)
   {
   fsm::reset(state);
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

// FSM operations (advance/output/step)

void ccbfsm::advance(int& input)
   {
   fsm::advance(input);
   bitfield ip = determineinput(input);
   bitfield sin = determinefeedin(ip);
   // Update input
   input = ip;
   // Compute next state
   for(int i=0; i<k; i++)
      reg(i) = sin[i] >> reg(i);
   }

int ccbfsm::output(int input) const
   {
   bitfield ip = determineinput(input);
   bitfield sin = determinefeedin(ip);
   // Compute output
   bitfield op(0,0);
   for(int j=0; j<n; j++)
      {
      bitfield thisop(0,1);
      for(int i=0; i<k; i++)
         thisop ^= (sin[i] + reg(i)) * gen(i,j);
      op = thisop + op;
      }
   return op;
   }

// description output

std::string ccbfsm::description() const
   {
   std::ostringstream sout;
   sout << "(nu=" << nu << ", rate " << k << "/" << n << ", G=[";
   for(int i=0; i<k; i++)
      for(int j=0; j<n; j++)
         sout << gen(i,j) << (j==n-1 ? (i==k-1 ? "])" : "; ") : ", ");
   return sout.str();
   }

// object serialization

std::ostream& ccbfsm::serialize(std::ostream& sout) const
   {
   sout << gen;
   return sout;
   }

std::istream& ccbfsm::serialize(std::istream& sin)
   {
   sin >> gen;
   init(gen);
   return sin;
   }

}; // end namespace
