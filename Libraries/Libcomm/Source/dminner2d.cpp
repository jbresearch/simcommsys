/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "dminner2d.h"
#include "timer.h"
#include <sstream>

namespace libcomm {

// description output

template <class real, bool normalize>
std::string dminner2d<real,normalize>::description() const
   {
   std::ostringstream sout;
   sout << "Iterative 2D DM Inner Code";
   return sout.str();
   }

// object serialization - saving

template <class real, bool normalize>
std::ostream& dminner2d<real,normalize>::serialize(std::ostream& sout) const
   {
   return sout;
   }

// object serialization - loading

template <class real, bool normalize>
std::istream& dminner2d<real,normalize>::serialize(std::istream& sin)
   {
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner2d<logrealfast,false>;
template <>
const serializer dminner2d<logrealfast,false>::shelper
   = serializer("blockmodem", "dminner2d<logrealfast>", dminner2d<logrealfast,false>::create);

template class dminner2d<double,true>;
template <>
const serializer dminner2d<double,true>::shelper
   = serializer("blockmodem", "dminner2d<double>", dminner2d<double,true>::create);

}; // end namespace
