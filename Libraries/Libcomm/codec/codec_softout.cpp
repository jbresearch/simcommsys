/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "codec_softout.h"
#include "hard_decision.h"

namespace libcomm {

template <class dbl>
void codec_softout<libbase::vector, dbl>::init_decoder(const array1vd_t& ptable)
   {
   array1vd_t temp;
   temp = ptable;
   this->setreceiver(temp);
   this->resetpriors();
   }

template <class dbl>
void codec_softout<libbase::vector, dbl>::init_decoder(
      const array1vd_t& ptable, const array1vd_t& app)
   {
   setreceiver(ptable);
   setpriors(app);
   }

template <class dbl>
void codec_softout<libbase::vector, dbl>::decode(array1i_t& decoded)
   {
   array1vd_t ri;
   softdecode(ri);
   hard_decision<libbase::vector, dbl> functor;
   functor(ri, decoded);
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"
#include "mpreal.h"

namespace libcomm {

using libbase::vector;
using libbase::logrealfast;
using libbase::mpreal;

template class codec_softout<vector, float> ;
template class codec_softout<vector, double> ;
template class codec_softout<vector, logrealfast> ;
template class codec_softout<vector, mpreal> ;

} // end namespace
