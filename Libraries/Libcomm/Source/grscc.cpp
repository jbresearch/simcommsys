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

using libbase::vector;
using libbase::matrix;


// Constructors / Destructors

/*! \brief Principal constructor
*/
template <class G> grscc<G>::grscc(const matrix< vector<G> >& generator) : ccfsm<G>(generator)
   {
   }

/*! \brief Copy constructor
*/
template <class G> grscc<G>::grscc(const grscc<G>& x) : ccfsm<G>(x)
   {
   }
   

// FSM state operations (getting and resetting)

/*! \brief Resets for circular trellis, given zero-state solution and number of time-steps
*/
template <class G> void grscc<G>::resetcircular(int zerostate, int n)
   {
   assert("Function not implemented.");
   }

/*! \brief Resets for circular trellis, assuming we have just run through the zero-state zero-input
*/
template <class G> void grscc<G>::resetcircular()
   {
   assert("Function not implemented.");
   }


// FSM helper operations

/*! \brief Determine the actual input that will be applied (resolve tail as necessary)
    \param  input    Requested input - can be any valid input or the special 'tail' value
    \return Either the given value, or the value that must be applied to tail out
   
    \warning I don't know why but GCC complains if I don't explicitly refer to member
             variables using this-> or grscc<G>::
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

/*! \brief Determine the value that will be shifted into the register
    \param  input    Requested input - can only be a valid input
    \return Vector representation of the shift-in value - lower index positions
            correspond to lower-index inputs
   
    \warning I don't know why but GCC complains if I don't explicitly refer to member
             variables using this-> or grscc<G>::
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

// object serialization - saving

template <class G> std::ostream& grscc<G>::serialize(std::ostream& sout) const
   {
   return ccfsm<G>::serialize(sout);
   }

// object serialization - loading

template <class G> std::istream& grscc<G>::serialize(std::istream& sin)
   {
   return ccfsm<G>::serialize(sin);
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

template class grscc< gf<4,0x13> >;
template <> const serializer grscc< gf<4,0x13> >::shelper = serializer("fsm", "grscc<gf<4,0x13>>", grscc< gf<4,0x13> >::create);

}; // end namespace
