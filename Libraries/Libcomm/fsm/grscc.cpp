/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
   trace << "DEBUG (grscc): period = " << L << "\n";
   // correspondence table has first index for N%L, second index for S_N^0
   csct.init(L, this->num_states());
   // go through all combinations (except N%L=0, which is illegal) and fill in
   Gi = eye;
   for (int i = 1; i < L; i++)
      {
      Gi *= stategen;
      const matrix<G> A = (eye + Gi).inverse();
      for (int j = 0; j < this->num_states(); j++)
         {
         vector<G> statevec = A * ccfsm<G>::convert(j, ccfsm<G>::nu);
         csct(i, j) = ccfsm<G>::convert(statevec);
         }
      }
   }

// FSM helper operations

template <class G>
vector<int> grscc<G>::determineinput(vector<int> input) const
   {
   for (int i = 0; i < input.size(); i++)
      if (input(i) == fsm::tail)
         {
         // Handle tailing out
         const G zero;
         for (int i = 0; i < this->k; i++)
            input(i) = convolve(zero, this->reg(i), this->gen(i, i));
         }
   return input;
   }

template <class G>
vector<G> grscc<G>::determinefeedin(vector<int> input) const
   {
   for (int i = 0; i < input.size(); i++)
      assert(input(i) != fsm::tail);
   // Determine the shift-in values by convolution
   vector<G> sin(this->k);
   for (int i = 0; i < this->k; i++)
      sin(i) = convolve(input(i), this->reg(i), this->gen(i, i));
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
void grscc<G>::resetcircular(vector<int> zerostate, int n)
   {
   // TODO: check the input state is valid
   assert(csct.size() > 0);
   const int L = csct.size().rows();
   assert(n % L != 0);
   const int zerostateval = ccfsm<G>::convert(vector<G>(zerostate));
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

// Explicit Realizations

#include "gf.h"

namespace libcomm {

using libbase::gf;
using libbase::serializer;

// Degenerate case GF(2)

template class grscc<gf<1, 0x3> > ;
template <>
const serializer grscc<gf<1, 0x3> >::shelper = serializer("fsm",
      "grscc<gf<1,0x3>>", grscc<gf<1, 0x3> >::create);

// cf. Lin & Costello, 2004, App. A

template class grscc<gf<2, 0x7> > ;
template <>
const serializer grscc<gf<2, 0x7> >::shelper = serializer("fsm",
      "grscc<gf<2,0x7>>", grscc<gf<2, 0x7> >::create);
template class grscc<gf<3, 0xB> > ;
template <>
const serializer grscc<gf<3, 0xB> >::shelper = serializer("fsm",
      "grscc<gf<3,0xB>>", grscc<gf<3, 0xB> >::create);
template class grscc<gf<4, 0x13> > ;
template <>
const serializer grscc<gf<4, 0x13> >::shelper = serializer("fsm",
      "grscc<gf<4,0x13>>", grscc<gf<4, 0x13> >::create);

template class grscc<gf<5, 0x25> > ;
template <>
const serializer grscc<gf<5, 0x25> >::shelper("fsm", "grscc<gf<5,0x25>>",
      grscc<gf<5, 0x25> >::create);

template class grscc<gf<6, 0x43> > ;
template <>
const serializer grscc<gf<6, 0x43> >::shelper("fsm", "grscc<gf<6,0x43>>",
      grscc<gf<6, 0x43> >::create);

template class grscc<gf<7, 0x89> > ;
template <>
const serializer grscc<gf<7, 0x89> >::shelper("fsm", "grscc<gf<7,0x89>>",
      grscc<gf<7, 0x89> >::create);

template class grscc<gf<8, 0x11D> > ;
template <>
const serializer grscc<gf<8, 0x11D> >::shelper("fsm", "grscc<gf<8,0x11D>>",
      grscc<gf<8, 0x11D> >::create);

template class grscc<gf<9, 0x211> > ;
template <>
const serializer grscc<gf<9, 0x211> >::shelper("fsm", "grscc<gf<9,0x211>>",
      grscc<gf<9, 0x211> >::create);

template class grscc<gf<10, 0x409> > ;
template <>
const serializer grscc<gf<10, 0x409> >::shelper("fsm", "grscc<gf<10,0x409>>",
      grscc<gf<10, 0x409> >::create);

} // end namespace
