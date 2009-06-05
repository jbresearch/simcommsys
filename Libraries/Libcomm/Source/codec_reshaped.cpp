/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "codec_reshaped.h"

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

}; // end namespace

// Explicit Realizations

#include "turbo.h"
#include "uncoded.h"

namespace libcomm {

using libbase::serializer;

/*** Turbo codes ***/

template class codec_reshaped< turbo<double> >;
template <>
const serializer codec_reshaped< turbo<double> >::shelper = serializer("codec", "codec_reshaped<turbo<double>>", codec_reshaped< turbo<double> >::create);

/*** Uncoded/repetition codes ***/

template class codec_reshaped< uncoded<double> >;
template <>
const serializer codec_reshaped< uncoded<double> >::shelper = serializer("codec", "codec_reshaped<uncoded<double>>", codec_reshaped< uncoded<double> >::create);

}; // end namespace
