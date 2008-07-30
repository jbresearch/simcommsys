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

template <class dbl>
class codec_softout : public codec {
public:
   /*! \name Codec operations */
   void decode(libbase::vector<int>& decoded);
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void decode(libbase::matrix<dbl>& ri) = 0;
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep
      \param[out] ro Likelihood table for output symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void decode(libbase::matrix<dbl>& ri, libbase::matrix<dbl>& ro) = 0;
   // @}

   /*! \name Codec helper functions */
   /*!
      \brief Hard decision on soft information
      \param[in]  ri       Likelihood table for input symbols at every timestep
      \param[out] decoded  Sequence of the most likely input symbols at every
                           timestep

      Decide which input sequence was most probable.
   */
   static void hard_decision(const libbase::matrix<dbl>& ri, libbase::vector<int>& decoded);
   // @}
};

}; // end namespace

#endif

