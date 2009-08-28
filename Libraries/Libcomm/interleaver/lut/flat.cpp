/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "flat.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// initialization

template <class real>
void flat<real>::init(const int tau)
   {
   this->lut.init(tau);
   for (int i = 0; i < tau; i++)
      this->lut(i) = i;
   }

// description output

template <class real>
std::string flat<real>::description() const
   {
   std::ostringstream sout;
   sout << "Flat Interleaver";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& flat<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& flat<real>::serialize(std::istream& sin)
   {
   int tau;
   sin >> libbase::eatcomments >> tau;
   init(tau);
   return sin;
   }

// Explicit instantiations

template class flat<float> ;
template <>
const libbase::serializer flat<float>::shelper("interleaver", "flat<float>",
      flat<float>::create);

template class flat<double> ;
template <>
const libbase::serializer flat<double>::shelper("interleaver", "flat<double>",
      flat<double>::create);

template class flat<libbase::logrealfast> ;
template <>
const libbase::serializer flat<libbase::logrealfast>::shelper("interleaver",
      "flat<logrealfast>", flat<libbase::logrealfast>::create);

} // end namespace
