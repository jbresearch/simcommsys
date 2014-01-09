/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ccbfsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::bitfield;

// initialization

void ccbfsm::init()
   {
   // copy automatically what we can
   k = gen.size().rows();
   n = gen.size().cols();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and
   // create shift registers
   reg.init(k);
   nu = 0;
   for (int i = 0; i < k; i++)
      {
      // assume with of register of input 'i' from its generator sequence for first output
      int m = gen(i, 0).size() - 1;
      reg(i).init(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for (int j = 1; j < n; j++)
         assertalways(gen(i, j).size() == m + 1);
      // update memory order
      if (m > ccbfsm::m)
         ccbfsm::m = m;
      }
   }

// constructors / destructors

ccbfsm::ccbfsm(const libbase::matrix<bitfield>& generator)
   {
   gen = generator;
   init();
   }

// FSM state operations (getting and resetting)

libbase::vector<int> ccbfsm::state() const
   {
   libbase::vector<int> state(nu);
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            state(j++) = reg(i)(t);
   assert(j == nu);
   return state;
   }

void ccbfsm::reset()
   {
   fsm::reset();
   reg = 0;
   }

void ccbfsm::reset(const libbase::vector<int>& state)
   {
   fsm::reset(state);
   assert(state.size() == nu);
   reg = 0;
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            reg(i) |= bitfield(state(j++) << t, reg(i).size());
   assert(j == nu);
   }

// FSM operations (advance/output/step)

void ccbfsm::advance(libbase::vector<int>& input)
   {
   fsm::advance(input);
   input = determineinput(input);
   bitfield sin = determinefeedin(input);
   // Compute next state
   for (int i = 0; i < k; i++)
      reg(i) = reg(i) << sin.extract(i);
   }

libbase::vector<int> ccbfsm::output(const libbase::vector<int>& input) const
   {
   libbase::vector<int> ip = determineinput(input);
   bitfield sin = determinefeedin(ip);
   // Compute output
   libbase::vector<int> op(n);
   for (int j = 0; j < n; j++)
      {
      bitfield thisop(0, 1);
      for (int i = 0; i < k; i++)
         thisop ^= (reg(i) + sin.extract(i)) * gen(i, j);
      op(j) = thisop;
      }
   return op;
   }

// description output

std::string ccbfsm::description() const
   {
   std::ostringstream sout;
   sout << "(nu=" << nu << ", rate " << k << "/" << n << ", G=[";
   for (int i = 0; i < k; i++)
      for (int j = 0; j < n; j++)
         sout << gen(i, j) << (j == n - 1 ? (i == k - 1 ? "])" : "; ") : ", ");
   return sout.str();
   }

// object serialization

std::ostream& ccbfsm::serialize(std::ostream& sout) const
   {
   sout << "#: Generator matrix (k x n bitfields)" << std::endl;
   sout << gen;
   return sout;
   }

std::istream& ccbfsm::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> gen >> libbase::verify;
   init();
   return sin;
   }

} // end namespace
