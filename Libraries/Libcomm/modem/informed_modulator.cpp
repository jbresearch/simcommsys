/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "informed_modulator.h"

#include "gf.h"

namespace libcomm {

// Block modem operations

template <class S, template <class > class C>
void informed_modulator<S, C>::demodulate(const channel<S, C>& chan,
      const C<S>& rx, const C<array1d_t>& app, C<array1d_t>& ptable)
   {
   this->advance_if_dirty();
   dodemodulate(chan, rx, app, ptable);
   this->mark_as_dirty();
   }

// Explicit Realizations

using libbase::gf;

template class informed_modulator<gf<1, 0x3> > ;
template class informed_modulator<gf<2, 0x7> > ;
template class informed_modulator<gf<3, 0xB> > ;
template class informed_modulator<gf<4, 0x13> > ;
template class informed_modulator<gf<5, 0x25> > ;
template class informed_modulator<gf<6, 0x43> > ;
template class informed_modulator<gf<7, 0x89> > ;
template class informed_modulator<gf<8, 0x11D> > ;
template class informed_modulator<gf<9, 0x211> > ;
template class informed_modulator<gf<10, 0x409> > ;
template class informed_modulator<bool> ;
template class informed_modulator<sigspace> ;

} // end namespace
