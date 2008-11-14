#ifndef __codec_softout_h
#define __codec_softout_h

#include "config.h"
#include "codec.h"

namespace libcomm {

/*!
   \brief   Channel Codec with Soft Output Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

template <class dbl, template<class> class C=libbase::vector>
class codec_softout_interface : public codec<C> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl>     array1d_t;
   // @}
public:
   /*! \name Codec operations */
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void softdecode(C<array1d_t>& ri) = 0;
   /*!
      \brief Decoding process
      \param[out] ri Likelihood table for input symbols at every timestep
      \param[out] ro Likelihood table for output symbols at every timestep

      \note Each call to decode will perform a single iteration (with respect
            to num_iter).
   */
   virtual void softdecode(C<array1d_t>& ri, C<array1d_t>& ro) = 0;
   // @}
};

/*!
   \brief   Channel Codec with Soft Output Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Templated soft-output codec base. This extra level is required to allow
   partial specialization of the container.
*/

template <class dbl, template<class> class C=libbase::vector>
class codec_softout : public codec_softout_interface<dbl,C> {
public:
};

/*!
   \brief   Channel Codec with Soft Output Base Specialization.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Templated soft-output codec base. This extra level is required to allow
   partial specialization of the container.
*/

template <class dbl>
class codec_softout<dbl,libbase::vector> : public codec_softout_interface<dbl,libbase::vector> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<dbl>        array1d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
public:
   // Codec operations
   void decode(array1i_t& decoded);

   /*! \name Codec helper functions */
   /*!
      \brief Hard decision on soft information
      \param[in]  ri       Likelihood table for input symbols at every timestep
      \param[out] decoded  Sequence of the most likely input symbols at every
                           timestep

      Decide which input sequence was most probable.
   */
   static void hard_decision(const array1vd_t& ri, array1i_t& decoded);
   // @}
};

}; // end namespace

#endif

