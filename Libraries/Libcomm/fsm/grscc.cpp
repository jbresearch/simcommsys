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
 *
 * \warning GCC complains if I don't explicitly refer to member variables from
 * parent class ccfm<G> using this-> or grscc<G>:: qualifiers. It turns
 * out this is a known "feature" of GCC:
 * (c.f. http://gcc.gnu.org/bugs.html#known)
 *
 * \code
 * # This also affects members of base classes, see [14.6.2]:
 *
 * template <typename> struct A
 * {
 * int i, j;
 * };
 *
 * template <typename T> struct B : A<T>
 * {
 * int foo1() { return i; }       // error
 * int foo2() { return this->i; } // OK
 * int foo3() { return B<T>::i; } // OK
 * int foo4() { return A<T>::i; } // OK
 *
 * using A<T>::j;
 * int foo5() { return j; }       // OK
 * };
 * \endcode
 */

#include "grscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::trace;
using libbase::vector;
using libbase::matrix;

// Internal functions

/*!
 * \brief Create state-generator matrix in the required format for
 * determining circulation state
 * \return State-generator matrix
 *
 * The size of state-generator matrix \f$ G \f$ is \f$ \nu \times \nu \f$
 * elements. Each row contains the multipliers corresponding to a particular
 * memory element's input. In turn, each column contains the multiplier
 * (weight) corresponding to successive present-state memory elements.
 *
 * Note that by definition, \f$ G \f$ contains only the taps corresponding to
 * the feedforward and feedback paths for the next-state generation; thus the
 * polynomials corresponding to the output generation have no bearing.
 * Similarly, the taps corresponding to the inputs also are irrelevant.
 */
template <class G>
matrix<G> grscc<G>::getstategen() const
   {
   // Create generator matrix in required format
   matrix<G> stategen(this->nu, this->nu);
   stategen = G(0);
   // Consider each input in turn
   for (int i = 0, row = 0; i < this->k; i++, row++)
      {
      // First row describes the shift-input taps, except for
      // the first element, which corresponds to the shift-in quantity
      for (int j = 1, col = 0; j < this->gen(i, i).size(); j++, col++)
         stategen(col, row) = this->gen(i, i)(j);
      // Successive rows describe the simple right-shift taps
      for (int j = 1, col = 0; j < this->reg(i).size(); j++, col++)
         stategen(j - 1, ++row) = 1;
      }
   trace << "DEBUG (grscc): state-generator matrix = " << stategen;
   return stategen;
   }

/*!
 * \brief Initialize circulation state correspondence table
 *
 * If the feedback polynomial is primitive, the system behaves as a maximal-length
 * linear feedback shift register. We verify the period by computing the necessary
 * powers of the state-generator matrix.
 */
template <class G>
void grscc<G>::initcsct()
   {
   const matrix<G> stategen = getstategen();
   const matrix<G> eye = matrix<G>::eye(this->nu);
   matrix<G> Gi;
   // determine period
   int L;
   Gi = stategen;
   for (L = 1; (eye + Gi).max() > 0; L++)
      Gi *= stategen;
   trace << "DEBUG (grscc): period = " << L << std::endl;
   // correspondence table has first index for N%L, second index for S_N^0
   csct.init(L, this->num_states());
   // go through all combinations (except N%L=0, which is illegal) and fill in
   Gi = eye;
   for (int i = 1; i < L; i++)
      {
      Gi *= stategen;
      const matrix<G> IGi = eye + Gi;
      // check if matrix is non-invertible
      if (IGi.rank() < IGi.size().rows())
         {
         // clear circulation table to indicate we cannot do this
         csct.init(0, 0);
         return;
         }
      // compute inverse and determine circulation values at this offset
      const matrix<G> A = IGi.inverse();
      for (int j = 0; j < this->num_states(); j++)
         {
         vector<G> statevec = A * ccfsm<G>::convert(j, ccfsm<G>::nu);
         csct(i, j) = ccfsm<G>::convert(statevec);
         }
      }
   }

// FSM helper operations

template <class G>
vector<int> grscc<G>::determineinput(const vector<int>& input) const
   {
   vector<int> ip = input;
   for (int i = 0; i < ip.size(); i++)
      if (ip(i) == fsm::tail)
         {
         // Handle tailing out
         const G zero;
         for (int i = 0; i < this->k; i++)
            ip(i) = this->convolve(zero, this->reg(i), this->gen(i, i));
         }
   return ip;
   }

template <class G>
vector<G> grscc<G>::determinefeedin(const vector<int>& input) const
   {
   for (int i = 0; i < input.size(); i++)
      assert(input(i) != fsm::tail);
   // Determine the shift-in values by convolution
   vector<G> sin(this->k);
   for (int i = 0; i < this->k; i++)
      sin(i) = this->convolve(input(i), this->reg(i), this->gen(i, i));
   return sin;
   }

// FSM state operations (getting and resetting)

/*!
 * \copydoc fsm::resetcircular()
 *
 * Consider a convolutional code where the state \f$ S_i \f$ at timestep
 * \f$ i \f$ is related to state \f$ S_{i-1} \f$ and input \f$ X_i \f$ by the
 * relation:
 * \f[ S_i = G \cdot S_{i-1} + X_i \f]
 *
 * Therefore, after \f$ N \f$ timesteps, the state is given by:
 * \f[ S_N = G^N \cdot S_0 + \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]
 *
 * Thus, the circulation state, defined such that \f$ S_c = S_N = S_0 \f$ is
 * derived from the equation:
 * \f[ S_c = \langle I + G^N \rangle ^{-1} \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]
 *
 * and is obtainable only if \f$ I + G^N \f$ is invertible. It is worth noting
 * that not all \f$ G \f$ matrices are suitable; also, the sequence length
 * \f$ N \f$ must not be a multiple of the period \f$ L \f$ of the recursive
 * generator, defined by \f$ G^L = I \f$.
 *
 * Consider starting at the zero-intial-state and pre-encoding the input
 * sequence; this gives us a final state:
 * \f[ S_N^0 = \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]
 *
 * Combining this with the equation for the circulation state, we get:
 * \f[ S_c = \langle I + G^N \rangle ^{-1} S_N^0 \f]
 *
 * Note, however, that because of the periodicity of the system, this equation
 * can be reduced to:
 * \f[ S_c = \langle I + G^P \rangle ^{-1} S_N^0 \f]
 *
 * where \f$ P = N \mathrm{mod} L \f$. This can be obtained by a lookup table
 * containing all combinations of \f$ P \f$ and \f$ S_N^0 \f$.
 */
template <class G>
void grscc<G>::resetcircular(const vector<int>& zerostate, int n)
   {
   // TODO: check the input state is valid
   assert(csct.size() > 0);
   const int L = csct.size().rows();
   assert(n % L != 0);
   const int zerostateval = ccfsm<G>::convert(vector<G> (zerostate));
   const int circstateval = csct(n % L, zerostateval);
   this->reset(vector<int> (ccfsm<G>::convert(circstateval, ccfsm<G>::nu)));
   }

// Description

template <class G>
std::string grscc<G>::description() const
   {
   std::ostringstream sout;
   sout << "RSC code " << ccfsm<G>::description();
   return sout.str();
   }

// Serialization Support

template <class G>
std::ostream& grscc<G>::serialize(std::ostream& sout) const
   {
   return ccfsm<G>::serialize(sout);
   }

template <class G>
std::istream& grscc<G>::serialize(std::istream& sin)
   {
   ccfsm<G>::serialize(sin);
   initcsct();
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

/* Serialization string: grscc<type>
 * where:
 *      type = gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
   template class grscc<type>; \
   template <> \
   const serializer grscc<type>::shelper( \
         "fsm", \
         "grscc<" BOOST_PP_STRINGIZE(type) ">", \
         grscc<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, GF_TYPE_SEQ)

} // end namespace
