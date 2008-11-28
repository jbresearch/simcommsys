#ifndef __commsys_iterative_h
#define __commsys_iterative_h

#include "commsys.h"

namespace libcomm {

/*!
   \brief   Iterative-Decoding Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Communication system with iterative demodulation.

   \todo Add this class to Windows solution

   \todo Add this class to serializer_libcomm

   \todo Update interface to cater for various iterative modes between codec
         and modem.

   \todo Move iterative codec nature into this class (requires codec split).
*/

template <class S, template<class> class C=libbase::vector>
class commsys_iterative : public commsys<S,C> {
private:
   /*! \name User parameters */
   int   iter;    //!< Number of demodulation iterations
   // @}
public:
   // Communication System Interface
   void translate(const libbase::vector<S>& received);

   // Description
   std::string description() const;
   // Serialization Support
   DECLARE_SERIALIZER(commsys_iterative);
};

}; // end namespace

#endif
