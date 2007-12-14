/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "gnrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;


// Constructors / Destructors

/*! \brief Principal constructor
*/
template <class G> gnrcc<G>::gnrcc(const matrix< vector<G> >& generator) : ccfsm<G>(generator)
   {
   }

/*! \brief Copy constructor
*/
template <class G> gnrcc<G>::gnrcc(const gnrcc<G>& x) : ccfsm<G>(x)
   {
   }
   

// FSM state operations (getting and resetting)

/*! \brief Resets for circular trellis, given zero-state solution and number of time-steps
*/
template <class G> void gnrcc<G>::resetcircular(int zerostate, int n)
   {
   assert("Function not implemented.");
   }

/*! \brief Resets for circular trellis, assuming we have just run through the zero-state zero-input
*/
template <class G> void gnrcc<G>::resetcircular()
   {
   assert("Function not implemented.");
   }


// FSM helper operations

/*! \brief Determine the actual input that will be applied (resolve tail as necessary)
    \param  input    Requested input - can be any valid input or the special 'tail' value
    \return Either the given value, or the value that must be applied to tail out
   
    \warning I don't know why but GCC complains if I don't explicitly refer to member
             variables using this-> or gnrcc<G>::
*/
template <class G> int gnrcc<G>::determineinput(int input) const
   {
   if(input != fsm::tail)
      return input;
   return 0;
   }

/*! \brief Determine the value that will be shifted into the register
    \param  input    Requested input - can only be a valid input
    \return Vector representation of the shift-in value - lower index positions
            correspond to lower-index inputs
   
    \warning I don't know why but GCC complains if I don't explicitly refer to member
             variables using this-> or gnrcc<G>::
*/
template <class G> vector<G> gnrcc<G>::determinefeedin(int input) const
   {
   assert(input != fsm::tail);
   // Convert input to vector representation
   vector<G> ip(this->k);
   convert(input, ip);
   return ip;
   }

// description output

template <class G> std::string gnrcc<G>::description() const
   {
   std::ostringstream sout;
   sout << "NRC code " << ccfsm<G>::description();
   return sout.str();
   }

// object serialization - saving

template <class G> std::ostream& gnrcc<G>::serialize(std::ostream& sout) const
   {
   return ccfsm<G>::serialize(sout);
   }

// object serialization - loading

template <class G> std::istream& gnrcc<G>::serialize(std::istream& sin)
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

// cf. Lin & Costello, 2004, App. A

template class gnrcc< gf<4,0x13> >;
template <> const serializer gnrcc< gf<4,0x13> >::shelper = serializer("fsm", "gnrcc<gf<4,0x13>>", gnrcc< gf<4,0x13> >::create);

}; // end namespace
