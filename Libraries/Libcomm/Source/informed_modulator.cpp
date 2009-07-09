/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "informed_modulator.h"

#include "gf.h"

namespace libcomm {

// Vector modem operations

template <class S, template <class > class C>
void informed_modulator<S, C>::demodulate(const channel<S, C>& chan,
      const C<S>& rx, const C<array1d_t>& app, C<array1d_t>& ptable)
   {
   this->advance_if_dirty();
   dodemodulate(chan, rx, app, ptable);
   this->mark_as_dirty();
   }

// Explicit Realizations

template class informed_modulator<sigspace> ;
template class informed_modulator<bool> ;
template class informed_modulator<libbase::gf<1, 0x3> > ;
template class informed_modulator<libbase::gf<2, 0x7> > ;
template class informed_modulator<libbase::gf<3, 0xB> > ;
template class informed_modulator<libbase::gf<4, 0x13> > ;
} // end namespace
