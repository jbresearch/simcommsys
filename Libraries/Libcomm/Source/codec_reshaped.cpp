/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "codec_reshaped.h"
#include "turbo.h"

namespace libcomm {

// object serialization - saving

template <class base_codec>
std::ostream& codec_reshaped<base_codec>::serialize(std::ostream& sout) const
   {
   return base.serialize(sout);
   }

// object serialization - loading

template <class base_codec>
std::istream& codec_reshaped<base_codec>::serialize(std::istream& sin)
   {
   return base.serialize(sin);
   }

// Explicit Realizations

using libbase::serializer;

template class codec_reshaped< turbo<double> >;
template <>
const serializer codec_reshaped< turbo<double> >::shelper = serializer("codec", "codec_reshaped<turbo<double>>", codec_reshaped< turbo<double> >::create);

}; // end namespace
