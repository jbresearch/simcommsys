/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \see grscc.cpp
*/

#include "gnrcc.h"
#include <iostream>
#include <sstream>

namespace libcomm {

using libbase::vector;
using libbase::matrix;


// FSM helper operations

template <class G>
int gnrcc<G>::determineinput(int input) const
   {
   if(input != fsm::tail)
      return input;
   return 0;
   }

template <class G>
vector<G> gnrcc<G>::determinefeedin(int input) const
   {
   assert(input != fsm::tail);
   // Convert input to vector representation
   vector<G> ip(this->k);
   convert(input, ip);
   return ip;
   }


// FSM state operations (getting and resetting)

template <class G>
void gnrcc<G>::resetcircular(int zerostate, int n)
   {
   assertalways("Function not implemented.");
   }


// Description

template <class G>
std::string gnrcc<G>::description() const
   {
   std::ostringstream sout;
   sout << "NRC code " << ccfsm<G>::description();
   return sout.str();
   }

// Serialization Support

template <class G>
std::ostream& gnrcc<G>::serialize(std::ostream& sout) const
   {
   return ccfsm<G>::serialize(sout);
   }

template <class G>
std::istream& gnrcc<G>::serialize(std::istream& sin)
   {
   return ccfsm<G>::serialize(sin);
   }

}; // end namespace

// Explicit Realizations

#include "gf.h"

namespace libcomm {

using libbase::gf;
using libbase::serializer;

// Degenerate case GF(2)

template class gnrcc< gf<1,0x3> >;
template <>
const serializer gnrcc< gf<1,0x3> >::shelper = serializer("fsm", "gnrcc<gf<1,0x3>>", gnrcc< gf<1,0x3> >::create);

// cf. Lin & Costello, 2004, App. A

template class gnrcc< gf<2,0x7> >;
template <>
const serializer gnrcc< gf<2,0x7> >::shelper = serializer("fsm", "gnrcc<gf<2,0x7>>", gnrcc< gf<2,0x7> >::create);
template class gnrcc< gf<3,0xB> >;
template <>
const serializer gnrcc< gf<3,0xB> >::shelper = serializer("fsm", "gnrcc<gf<3,0xB>>", gnrcc< gf<3,0xB> >::create);
template class gnrcc< gf<4,0x13> >;
template <>
const serializer gnrcc< gf<4,0x13> >::shelper = serializer("fsm", "gnrcc<gf<4,0x13>>", gnrcc< gf<4,0x13> >::create);

}; // end namespace
