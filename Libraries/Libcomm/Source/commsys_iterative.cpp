/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_iterative.h"

#include "gf.h"

namespace libcomm {


// Explicit Realizations

template class commsys_iterative<sigspace>;
template <>
const libbase::serializer commsys_iterative<sigspace>::shelper("commsys", "commsys_iterative<sigspace>", commsys_iterative<sigspace>::create);

template class commsys_iterative<bool>;
template <>
const libbase::serializer commsys_iterative<bool>::shelper("commsys", "commsys_iterative<bool>", commsys_iterative<bool>::create);
template class commsys_iterative< libbase::gf<1,0x3> >;
template <>
const libbase::serializer commsys_iterative< libbase::gf<1,0x3> >::shelper("commsys", "commsys_iterative<gf<1,0x3>>", commsys_iterative< libbase::gf<1,0x3> >::create);
template class commsys_iterative< libbase::gf<2,0x7> >;
template <>
const libbase::serializer commsys_iterative< libbase::gf<2,0x7> >::shelper("commsys", "commsys_iterative<gf<2,0x7>>", commsys_iterative< libbase::gf<2,0x7> >::create);
template class commsys_iterative< libbase::gf<3,0xB> >;
template <>
const libbase::serializer commsys_iterative< libbase::gf<3,0xB> >::shelper("commsys", "commsys_iterative<gf<3,0xB>>", commsys_iterative< libbase::gf<3,0xB> >::create);
template class commsys_iterative< libbase::gf<4,0x13> >;
template <>
const libbase::serializer commsys_iterative< libbase::gf<4,0x13> >::shelper("commsys", "commsys_iterative<gf<4,0x13>>", commsys_iterative< libbase::gf<4,0x13> >::create);

}; // end namespace
