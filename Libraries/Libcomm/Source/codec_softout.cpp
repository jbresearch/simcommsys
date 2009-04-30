/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "codec_softout.h"

namespace libcomm {

template <class dbl>
void codec_softout<dbl,libbase::vector>::translate(const libbase::vector< libbase::vector<double> >& ptable)
   {
   array1vd_t temp;
   temp = ptable;
   this->setreceiver(temp);
   this->resetpriors();
   }

template <class dbl>
void codec_softout<dbl,libbase::vector>::translate(const array1vd_t& ptable, const array1vd_t& app)
   {
   setreceiver(ptable);
   setpriors(app);
   }

template <class dbl>
void codec_softout<dbl,libbase::vector>::decode(array1i_t& decoded)
   {
   array1vd_t ri;
   softdecode(ri);
   hard_decision(ri, decoded);
   }

template <class dbl>
void codec_softout<dbl,libbase::vector>::hard_decision(const array1vd_t& ri, array1i_t& decoded)
   {
   // Determine sizes from input matrix
   const int tau = ri.size();
   assert(tau > 0);
   const int K = ri(0).size();
   // Initialise result vector
   decoded.init(tau);
   // Determine most likely symbol at every timestep
   for(int t=0; t<tau; t++)
      {
      assert(ri(t).size() == K);
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(ri(t)(i) > ri(t)(decoded(t)))
            decoded(t) = i;
      }
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

template class codec_softout<float>;
template class codec_softout<double>;
template class codec_softout<libbase::logrealfast>;

}; // end namespace
