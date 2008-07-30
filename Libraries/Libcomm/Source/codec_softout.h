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
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~codec_softout() {};
   // @}

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
};

}; // end namespace

#endif

