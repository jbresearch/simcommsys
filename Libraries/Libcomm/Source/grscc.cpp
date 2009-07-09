/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$

 \warning GCC complains if I don't explicitly refer to member variables from
 parent class ccfm<G> using this-> or grscc<G>:: qualifiers. It turns
 out this is a known "feature" of GCC:
 (c.f. http://gcc.gnu.org/bugs.html#known)

 \code
 # This also affects members of base classes, see [14.6.2]:

 template <typename> struct A
 {
 int i, j;
 };

 template <typename T> struct B : A<T>
 {
 int foo1() { return i; }       // error
 int foo2() { return this->i; } // OK
 int foo3() { return B<T>::i; } // OK
 int foo4() { return A<T>::i; } // OK

 using A<T>::j;
 int foo5() { return j; }       // OK
 };
 \endcode
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
 \brief Determine unique value from state vector
 \param statevec State vector in the required format for determining circulation state
 \return Unique integer representation of state value

 Similarly to convention, define the state vector as a column vector, as follows:
 \f[ S_i = \begin{pmatrix}
 S_{1,1} \\ S_{2,1} \\ \vdots \\ S_{\nu_1,1} \\
                  S_{1,2} \\ S_{2,2} \\ \vdots \\ S_{\nu_2,2} \\
                  \vdots \\ S_{\nu_k,k}
 \end{pmatrix} \f]

 where \f$ k \f$ is the number of inputs and \f$ \nu_i \f$ is the number of
 memory elements for input \f$ i \f$. Note that conventionally, element \f$ S_{1,i} \f$
 is the left-most memory element for input \f$ i \f$, and therefore the one to which
 the shift-in is applied. It can be seen that the total length of the state vector
 is equal to the total number of memory elements in the system, \f$ \nu \f$.
 */
template <class G>
int grscc<G>::getstateval(const vector<G>& statevec) const
   {
   int stateval = 0;
   for (int i = 0; i < this->nu; i++)
      {
      stateval *= G::elements();
      stateval += statevec(i);
      }
   assert(stateval >= 0 && stateval < this->num_states());
   //trace << "DEBUG (grscc): state value = " << stateval << "\n";
   return stateval;
   }

/*!
 \brief Convert integer representation of state value to a vector in the required
 format for determining circulation state
 \param stateval Unique integer representation of state value
 \return State vector in the required format for determining circulation state
 */
template <class G>
vector<G> grscc<G>::getstatevec(int stateval) const
   {
   // Create generator matrix in required format
   vector<G> statevec(this->nu);
   for (int i = this->nu - 1; i >= 0; i--)
      {
      statevec(i) = stateval % G::elements();
      stateval /= G::elements();
      }
   assert(stateval == 0);
   //trace << "DEBUG (grscc): state vector = ";
   //statevec.serialize(trace);
   return statevec;
   }

/*!
 \brief Create state-generator matrix in the required format for
 determining circulation state
 \return State-generator matrix

 The size of state-generator matrix \f$ G \f$ is \f$ \nu \times \nu \f$ elements.
 Each row contains the multipliers corresponding to a particular memory element's
 input. In turn, each column contains the multiplier (weight) corresponding to
 successive present-state memory elements.

 Note that by definition, \f$ G \f$ contains only the taps corresponding to the
 feedforward and feedback paths for the next-state generation; thus the polynomials
 corresponding to the output generation have no bearing. Similarly, the taps
 corresponding to the inputs also are irrelevant.
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
      for (int j = this->gen(i, i).size() - 2, col = 0; j >= 0; j--, col++)
         stategen(col, row) = this->gen(i, i)(j);
      // Successive rows describe the simple right-shift taps
      for (int j = 1; j < this->reg(i).size(); j++)
         stategen(j - 1, ++row) = 1;
      }
   trace << "DEBUG (grscc): state-generator matrix = " << stategen;
   return stategen;
   }

/*!
 \brief Initialize circulation state correspondence table

 If the feedback polynomial is primitive, the system behaves as a maximal-length
 linear feedback shift register. We verify the period by computing the necessary
 powers of the state-generator matrix.
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
         vector<G> statevec = A * getstatevec(j);
         csct(i, j) = getstateval(statevec);
         }
      }
   }

// FSM helper operations

template <class G>
int grscc<G>::determineinput(int input) const
   {
   if (input != fsm::tail)
      return input;
   // Handle tailing out
   const G zero;
   vector<G> ip(this->k);
   for (int i = 0; i < this->k; i++)
      ip(i) = convolve(zero, this->reg(i), this->gen(i, i));
   return convert(ip);
   }

template <class G>
vector<G> grscc<G>::determinefeedin(int input) const
   {
   assert(input != fsm::tail);
   // Convert input to vector representation
   vector<G> ip(this->k);
   convert(input, ip);
   // Determine the shift-in values by convolution
   vector<G> sin(this->k);
   for (int i = 0; i < this->k; i++)
      sin(i) = convolve(ip(i), this->reg(i), this->gen(i, i));
   return sin;
   }

// FSM state operations (getting and resetting)

template <class G>
void grscc<G>::resetcircular(int zerostate, int n)
   {
   assert(zerostate >= 0 && zerostate < this->num_states());
   if (csct.size() == 0)
      initcsct();
   const int L = csct.size().rows();
   assert(n%L != 0);
   reset(csct(n % L, zerostate));
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
   return ccfsm<G>::serialize(sin);
   }

} // end namespace

// Explicit Realizations

#include "gf.h"

namespace libcomm {

using libbase::gf;
using libbase::serializer;

// Degenerate case GF(2)

template class grscc<gf<1, 0x3> >
template <>
const serializer grscc<gf<1, 0x3> >::shelper = serializer("fsm",
      "grscc<gf<1,0x3>>", grscc<gf<1, 0x3> >::create);

// cf. Lin & Costello, 2004, App. A

template class grscc<gf<2, 0x7> >
template <>
const serializer grscc<gf<2, 0x7> >::shelper = serializer("fsm",
      "grscc<gf<2,0x7>>", grscc<gf<2, 0x7> >::create);
template class grscc<gf<3, 0xB> >
template <>
const serializer grscc<gf<3, 0xB> >::shelper = serializer("fsm",
      "grscc<gf<3,0xB>>", grscc<gf<3, 0xB> >::create);
template class grscc<gf<4, 0x13> >
template <>
const serializer grscc<gf<4, 0x13> >::shelper = serializer("fsm",
      "grscc<gf<4,0x13>>", grscc<gf<4, 0x13> >::create);

} // end namespace
