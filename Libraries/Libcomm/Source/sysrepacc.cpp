/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "sysrepacc.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// encoding and decoding functions

template <class real, class dbl>
void sysrepacc<real,dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   }

template <class real, class dbl>
void sysrepacc<real,dbl>::translate(const libbase::vector< libbase::vector<double> >& ptable)
   {
   }

template <class real, class dbl>
void sysrepacc<real,dbl>::softdecode(array1vd_t& ri)
   {
   }

template <class real, class dbl>
void sysrepacc<real,dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   assertalways("Not yet implemented");
   }

// description output

template <class real, class dbl>
std::string sysrepacc<real,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Systematic " << repacc<real,dbl>::description();
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& sysrepacc<real,dbl>::serialize(std::ostream& sout) const
   {
   return repacc<real,dbl>::serialize(sout);
   }

// object serialization - loading

template <class real, class dbl>
std::istream& sysrepacc<real,dbl>::serialize(std::istream& sin)
   {
   return repacc<real,dbl>::serialize(sin);
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
using libbase::serializer;

template class sysrepacc<double>;
template <>
const serializer sysrepacc<double>::shelper = serializer("codec", "sysrepacc<double>", sysrepacc<double>::create);

template class sysrepacc<logrealfast>;
template <>
const serializer sysrepacc<logrealfast>::shelper = serializer("codec", "sysrepacc<logrealfast>", sysrepacc<logrealfast>::create);

template class sysrepacc<logrealfast,logrealfast>;
template <>
const serializer sysrepacc<logrealfast,logrealfast>::shelper = serializer("codec", "sysrepacc<logrealfast,logrealfast>", sysrepacc<logrealfast,logrealfast>::create);

}; // end namespace
