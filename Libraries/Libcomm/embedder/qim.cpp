/*!
 * \file
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "qim.h"
#include "itfunc.h"
#include <cmath>
#include <sstream>

namespace libcomm {

using libbase::serializer;

// description output

template <class S>
std::string qim<S>::description() const
   {
   std::ostringstream sout;
   if (alpha < 1.0)
      sout << "DC-";
   sout << "QIM embedder (M=" << M << ", delta=" << delta;
   if (alpha < 1.0)
      sout << ", alpha=" << alpha;
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class S>
std::ostream& qim<S>::serialize(std::ostream& sout) const
   {
   sout << M << std::endl;
   sout << delta << std::endl;
   sout << alpha << std::endl;
   return sout;
   }

// object serialization - loading

template <class S>
std::istream& qim<S>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> M;
   sin >> libbase::eatcomments >> delta;
   sin >> libbase::eatcomments >> alpha;
   return sin;
   }

// Explicit Realizations

template class qim<int>;
template <>
const serializer qim<int>::shelper("embedder", "qim<int>", qim<int>::create);

template class qim<float>;
template <>
const serializer qim<float>::shelper("embedder", "qim<float>", qim<float>::create);

template class qim<double>;
template <>
const serializer qim<double>::shelper("embedder", "qim<double>", qim<double>::create);

} // end namespace
