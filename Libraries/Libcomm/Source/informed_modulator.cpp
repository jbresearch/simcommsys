/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "informed_modulator.h"

#include "gf.h"

namespace libcomm {

// Vector modem operations

template <class S>
void informed_modulator<S>::demodulate(const channel<S>& chan, const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   this->advance_if_dirty();
   dodemodulate(chan, rx, app, ptable);
   this->mark_as_dirty();
   }

// Explicit Realizations

template class informed_modulator<sigspace>;
template class informed_modulator<bool>;
template class informed_modulator< libbase::gf<1,0x3> >;
template class informed_modulator< libbase::gf<2,0x7> >;
template class informed_modulator< libbase::gf<3,0xB> >;
template class informed_modulator< libbase::gf<4,0x13> >;

}; // end namespace
