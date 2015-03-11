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

#include "ccfsm.h"
#include <iostream>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Debug advance() and output()
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

using libbase::trace;
using libbase::vector;
using libbase::matrix;

// Internal functions

/*!
 * \brief Initialization
 * \param  generator   Generator matrix of size \f$ k \times n \f$
 *
 * Each generator matrix element is a vector over G, laid out in the same format
 * as the internal registers - lower index positions are considered to lie on
 * the left, and correspond with register positions closest to the input
 * junction. This follows the usual convention in the coding community.
 */
template <class G>
void ccfsm<G>::init(const matrix<vector<G> >& generator)
   {
   // copy automatically what we can
   gen = generator;
   k = gen.size().rows();
   n = gen.size().cols();
   // set default value to the rest
   m = 0;
   // check that the generator matrix is valid (correct sizes) and create
   // shift registers
   reg.init(k);
   nu = 0;
   for (int i = 0; i < k; i++)
      {
      // assume width of register of input 'i' from its generator sequence for
      // first output
      int m = gen(i, 0).size() - 1;
      reg(i).init(m);
      nu += m;
      // check that the gen. seq. for all outputs are the same length
      for (int j = 1; j < n; j++)
         assertalways(gen(i, j).size() == m + 1);
      // update memory order
      if (m > ccfsm<G>::m)
         ccfsm<G>::m = m;
      }
   }

// Helper functions

/*!
 * \brief Convolves the shift-in value and register with a generator polynomial
 * \param  s  The value at the left shift-in of the register
 * \param  r  The register
 * \param  g  The corresponding generator polynomial
 * \return The output
 *
 * \todo Document this function with a diagram.
 */
template <class G>
G ccfsm<G>::convolve(const G& s, const vector<G>& r, const vector<G>& g) const
   {
   // Inherit sizes
   const int m = r.size();
   assert(g.size() == m + 1);
   // Convolve the shift-in value with corresponding generator polynomial
   G thisop = s * g(0);
   // Convolve register with corresponding generator polynomial
   for (int i = 0; i < m; i++)
      thisop += r(i) * g(i + 1);
   return thisop;
   }

// FSM state operations (getting and resetting)

template <class G>
vector<int> ccfsm<G>::state() const
   {
   vector<int> state(nu);
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            state(j++) = reg(i)(t);
   assert(j == nu);
   return state;
   }

template <class G>
void ccfsm<G>::reset(const vector<int>& state)
   {
   fsm::reset(state);
   assert(state.size() == nu);
   int j = 0;
   for (int t = 0; t < nu; t++)
      for (int i = 0; i < k; i++)
         if (reg(i).size() > t)
            reg(i)(t) = state(j++);
   assert(j == nu);
   }

// FSM operations (advance/output/step)

template <class G>
void ccfsm<G>::advance(vector<int>& input)
   {
   fsm::advance(input);
#if DEBUG>=2
   trace << "Advance:" << std::endl;
   trace << "  Original Input:\t";
   input.serialize(trace);
#endif
   input = determineinput(input);
#if DEBUG>=2
   trace << "  Actual Input: \t";
   input.serialize(trace);
#endif
   vector<G> sin = determinefeedin(input);
#if DEBUG>=2
   trace << "  Register Feed-in:\t";
   sin.serialize(trace);
#endif
   // Compute next state for each input register
   for (int i = 0; i < k; i++)
      {
#if DEBUG>=2
      trace << "  Register " << i << " In:\t";
      reg(i).serialize(trace);
#endif
      const int m = reg(i).size();
      if (m == 0)
         continue;
      // Shift entries to the right (ie. up)
      for (int j = m - 1; j > 0; j--)
         reg(i)(j) = reg(i)(j - 1);
      // Left-most entry gets the shift-in value
      reg(i)(0) = sin(i);
#if DEBUG>=2
      trace << "  Register " << i << " Out:\t";
      reg(i).serialize(trace);
#endif
      }
   }

template <class G>
vector<int> ccfsm<G>::output(const vector<int>& input) const
   {
#if DEBUG>=2
   trace << "Output:" << std::endl;
   trace << "  Original Input:\t";
   input.serialize(trace);
#endif
   vector<int> ip = determineinput(input);
#if DEBUG>=2
   trace << "  Actual Input: \t";
   input.serialize(trace);
#endif
   vector<G> sin = determinefeedin(ip);
#if DEBUG>=2
   trace << "  Register Feed-in:\t";
   sin.serialize(trace);
#endif
   // Compute output
   vector<G> op(n);
   for (int j = 0; j < n; j++)
      {
      G thisop;
      for (int i = 0; i < k; i++)
         {
         thisop += convolve(sin(i), reg(i), gen(i, j));
#if DEBUG>=2
         trace << "  Input + Register " << i << ":\t";
         trace << sin(i) << "\t";
         reg(i).serialize(trace);
         trace << "  Generator " << i << "," << j << ":\t";
         gen(i, j).serialize(trace);
         trace << "  Accumulated Result:\t" << thisop << std::endl;
#endif
         }
      op(j) = thisop;
      }
#if DEBUG>=2
   trace << "  Output:\t";
   op.serialize(trace);
#endif
   return vector<int> (op);
   }

// Description & Serialization

//! Description output - common part only, must be preceded by specific name
template <class G>
std::string ccfsm<G>::description() const
   {
   std::ostringstream sout;
   sout << "GF(" << G::elements() << "): (nu=" << nu << ", rate " << k << "/"
         << n << ", G=[";
   // Loop over all generator matrix elements
   for (int i = 0; i < k; i++)
      for (int j = 0; j < n; j++)
         {
         // Loop over polynomial
         for (int x = 0; x < gen(i, j).size(); x++)
            sout << "{" << gen(i, j)(x) << "}";
         sout << (j == n - 1 ? (i == k - 1 ? "])" : "; ") : ", ");
         }
   return sout.str();
   }

template <class G>
std::ostream& ccfsm<G>::serialize(std::ostream& sout) const
   {
   sout << "#: Generator matrix (k x n vectors)" << std::endl;
   sout << gen;
   return sout;
   }

template <class G>
std::istream& ccfsm<G>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> gen >> libbase::verify;
   init(gen);
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define INSTANTIATE(r, x, type) \
   template class ccfsm<type>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, GF_TYPE_SEQ)

} // end namespace
