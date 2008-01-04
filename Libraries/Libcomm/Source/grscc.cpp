/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "grscc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::trace;
using libbase::vector;
using libbase::matrix;


// FSM state operations (getting and resetting)

/*!
   \copydoc fsm::resetcircular(int zerostate, int n)

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

   Consequently, the size of state-generator matrix \f$ G \f$ is
   \f$ \nu \times \nu \f$ elements. Each row contains the multipliers corresponding
   to a particular memory element's input. In turn, each column contains the multiplier
   (weight) corresponding to successive present-state memory elements.

   Note that by definition, \f$ G \f$ contains only the taps corresponding to the
   feedforward and feedback paths for the next-state generation; thus the polynomials
   corresponding to the output generation have no bearing. Similarly, the taps
   corresponding to the inputs also are irrelevant.

   \warning I don't know why but GCC complains if I don't explicitly refer to member
            variables from parent class ccfm<G> using this-> or grscc<G>::
*/
template <class G> void grscc<G>::resetcircular(int zerostate, int n)
   {
   assert(zerostate >= 0 && zerostate < this->num_states());
   // Create generator matrix in required format
   matrix<G> stategen(this->nu,this->nu);
   stategen = G(libbase::int32u(0));
   // Consider each input in turn
   for(int i=0, row=0; i<this->k; i++, row++)
      {
      // First row describes the shift-input taps
      for(int j=this->gen(i,i).size()-1, col=0; j>=0; j--, col++)
         stategen(col,row) = this->gen(i,i)(j);
      // Successive rows describe the simple right-shift taps
      for(int j=1; j<this->reg(i).size(); j++)
         stategen(j-1,++row) = 1;
      }
   trace << "DEBUG (grscc): state-generator matrix = \n";
   stategen.serialize(trace);
   //assert(n%7 != 0);
   //reset(csct[n%7][zerostate]);
   assertalways("Function not implemented.");
   }


// FSM helper operations

/*!
   \brief Determine the actual input that will be applied (resolve tail as necessary)
   \param  input    Requested input - can be any valid input or the special 'tail' value
   \return Either the given value, or the value that must be applied to tail out

   \warning I don't know why but GCC complains if I don't explicitly refer to member
            variables from parent class ccfm<G> using this-> or grscc<G>::
*/
template <class G> int grscc<G>::determineinput(int input) const
   {
   if(input != fsm::tail)
      return input;
   // Handle tailing out
   const G zero;
   vector<G> ip(this->k);
   for(int i=0; i<this->k; i++)
      ip(i) = convolve(zero, this->reg(i), this->gen(i,i));
   return convert(ip);
   }

/*!
   \brief Determine the value that will be shifted into the register
   \param  input    Requested input - can only be a valid input
   \return Vector representation of the shift-in value - lower index positions
           correspond to lower-index inputs

   \warning I don't know why but GCC complains if I don't explicitly refer to member
            variables from parent class ccfm<G> using this-> or grscc<G>::
*/
template <class G> vector<G> grscc<G>::determinefeedin(int input) const
   {
   assert(input != fsm::tail);
   // Convert input to vector representation
   vector<G> ip(this->k);
   convert(input, ip);
   // Determine the shift-in values by convolution
   vector<G> sin(this->k);
   for(int i=0; i<this->k; i++)
      sin(i) = convolve(ip(i), this->reg(i), this->gen(i,i));
   return sin;
   }

// description output

template <class G> std::string grscc<G>::description() const
   {
   std::ostringstream sout;
   sout << "RSC code " << ccfsm<G>::description();
   return sout.str();
   }

}; // end namespace

// Explicit Realizations

#include "gf.h"
#include "serializer.h"

namespace libcomm {

using libbase::gf;
using libbase::serializer;

// Degenerate case GF(2)

template class grscc< gf<1,0x3> >;
template <> const serializer grscc< gf<1,0x3> >::shelper = serializer("fsm", "grscc<gf<1,0x3>>", grscc< gf<1,0x3> >::create);

// cf. Lin & Costello, 2004, App. A

template class grscc< gf<2,0x7> >;
template <> const serializer grscc< gf<2,0x7> >::shelper = serializer("fsm", "grscc<gf<2,0x7>>", grscc< gf<2,0x7> >::create);
template class grscc< gf<3,0xB> >;
template <> const serializer grscc< gf<3,0xB> >::shelper = serializer("fsm", "grscc<gf<3,0xB>>", grscc< gf<3,0xB> >::create);
template class grscc< gf<4,0x13> >;
template <> const serializer grscc< gf<4,0x13> >::shelper = serializer("fsm", "grscc<gf<4,0x13>>", grscc< gf<4,0x13> >::create);

}; // end namespace
