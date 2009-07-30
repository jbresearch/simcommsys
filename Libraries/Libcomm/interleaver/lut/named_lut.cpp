/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "named_lut.h"
#include <sstream>

namespace libcomm {

// description output

template <class real>
std::string named_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Named Interleaver (" << lutname;
   if (m > 0)
      sout << ", Forced tail length " << m << ")";
   else
      sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& named_lut<real>::serialize(std::ostream& sout) const
   {
   sout << m << "\n";
   sout << lutname << "\n";
   sout << this->lut;
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& named_lut<real>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> m;
   sin >> libbase::eatcomments >> lutname;
   sin >> libbase::eatcomments >> this->lut;
   return sin;
   }

// Explicit instantiations

template class named_lut<float> ;
template <>
const libbase::serializer named_lut<float>::shelper("interleaver",
      "named_lut<float>", named_lut<float>::create);

template class named_lut<double> ;
template <>
const libbase::serializer named_lut<double>::shelper("interleaver",
      "named_lut<double>", named_lut<double>::create);

template class named_lut<libbase::logrealfast> ;
template <>
const libbase::serializer named_lut<libbase::logrealfast>::shelper(
      "interleaver", "named_lut<logrealfast>",
      named_lut<libbase::logrealfast>::create);

} // end namespace
