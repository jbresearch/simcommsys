/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "codec_softout.h"

namespace libcomm {

template <class dbl>
void codec_softout<dbl,libbase::vector>::decode(libbase::vector<int>& decoded)
   {
   libbase::vector< libbase::vector<dbl> > ri;
   softdecode(ri);
   hard_decision(ri, decoded);
   }

template <class dbl>
void codec_softout<dbl,libbase::vector>::hard_decision(const libbase::vector< libbase::vector<dbl> >& ri, libbase::vector<int>& decoded)
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

template class codec_softout<double>;
template class codec_softout<libbase::logrealfast>;

}; // end namespace
