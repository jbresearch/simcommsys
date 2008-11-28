/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "informed_modulator.h"

namespace libcomm {

// Vector modem operations

template <class S>
void informed_modulator<S>::demodulate(const channel<S>& chan, const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   this->advance_if_dirty();
   dodemodulate(chan, rx, app, ptable);
   this->mark_as_dirty();
   }

}; // end namespace
