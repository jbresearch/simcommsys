#ifndef __codec_softout_h
#define __codec_softout_h

#include "config.h"
#include "codec.h"

namespace libcomm {

/*!
   \brief   Channel Codec with Soft Output.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

class codec_softout : public codec {
public:
   /*! \name Codec operations */
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void decode(libbase::matrix<double>& ri) = 0;
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep
      \param[out] ro Likelihood table for output symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void decode(libbase::matrix<double>& ri, libbase::matrix<double>& ro) = 0;
   // @}

   /*! \name Codec helper functions */
   /*!
      \brief Hard decision on soft information
      \param[in]  ri       Likelihood table for input symbols at every timestep
      \param[out] decoded  Sequence of the most likely input symbols at every
                           timestep

      Decide which input sequence was most probable.
   */
   template <class dbl>
   static void hard_decision(const libbase::matrix<dbl>& ri, libbase::vector<int>& decoded);
   // @}
};

// Templated functions

template <class dbl>
void codec_softout::hard_decision(const libbase::matrix<dbl>& ri, libbase::vector<int>& decoded)
   {
   // Determine sizes from input matrix
   const int tau = ri.xsize();
   const int K = ri.ysize();
   // Initialise result vector
   decoded.init(tau);
   // Determine most likely symbol at every timestep   
   for(int t=0; t<tau; t++)
      {
      decoded(t) = 0;
      for(int i=1; i<K; i++)
         if(ri(t, i) > ri(t, decoded(t)))
            decoded(t) = i;
      }
   }

}; // end namespace

#endif

